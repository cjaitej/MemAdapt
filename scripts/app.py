"""Streamlit UI for poking at a trained AMT checkpoint.

    streamlit run scripts/app.py

The same three traps as scripts/infer.py apply here, and are handled the same way:
router thresholds need calibrating before causal routing means anything, generation
never writes to the memory bank so it must be pre-warmed to observe the memory half,
and the vocab is padded past what GPT-2 BPE can decode.

What this shows that a terminal cannot
--------------------------------------
The point of the model is *which* tokens get depth and *which* tokens query memory,
so the samples are rendered with that decision attached to each token: background
weight is the number of adaptive layers the token passed through, an underline marks
a token that queried the bank. An aggregate routing rate cannot show that a token was
routed *because* its retrieval landed -- the per-token view can, and that is
contribution C1 made visible.

Sampling deliberately calls `AMT.generate` one token at a time rather than
reimplementing the loop. There is no KV cache in this model -- `generate` re-runs the
forward over the whole context every step regardless -- so a 1-token call costs
exactly what one iteration of its own loop costs, and the sampling logic stays in one
place. That per-token call is also what makes the routing readable: each call is
exactly one forward, so a forward hook fires exactly once per generated token and the
decision it records belongs to that token alone. The `torch.Generator` is created once
per sample and threaded through, or every token would be drawn from the same seed.

Results live in `st.session_state`, not in the `if generate:` branch. Streamlit reruns
the whole script on every widget interaction, so results computed inline vanish the
moment a slider moves -- which on a 4GB card means re-running a minute of sampling to
get back what was already on screen. Storing the decoded chunks and re-rendering from
them also makes the routing overlay a pure display toggle.
"""

import glob
import html
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import streamlit as st
import torch

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT, FlopModel  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.blocks import MemoryBlock  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import select_precision  # noqa: E402

GPT2_VOCAB = 50257
EOT = 50256       # <|endoftext|>; prepare.py prefixes every document with it
RAMP = 4          # background weight steps in the depth overlay
OVERRUN = 48      # tokens generation may run past its budget to finish a sentence

# Comparing more than a few checkpoints at once is a VRAM decision, not a layout one:
# every selected model stays resident (they are re-read on each rerun otherwise), and
# four 40M-parameter models plus one model's activations is already most of a 4GB
# card. The column width at four is also about as narrow as prose stays readable.
MAX_MODELS = 4
# Page width per column count. The single-model view keeps the old 52rem measure.
WIDTHS = {1: 52, 2: 68, 3: 86, 4: 104}

st.set_page_config(page_title="AMT playground", page_icon="🧠", layout="wide")

# Styling stays theme-agnostic: opacity, currentColor and colour-mixes rather than
# fixed colours, so it reads correctly whether the viewer is in Streamlit's light or
# dark theme. Every colour-mix is preceded by a flat fallback for browsers without it.
st.markdown("""
<style>
  /* max-width is injected once the column count is known -- see WIDTHS. */
  .block-container { padding-top: 2.5rem; margin: 0 auto; }
  .model-name {
    font-size: .72rem; opacity: .5; margin-bottom: .35rem;
    overflow-wrap: anywhere;
  }
  .chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: -.5rem 0 1.4rem; }
  .chip {
    font-size: .74rem; padding: .18rem .55rem; border-radius: 1rem;
    border: 1px solid currentColor; opacity: .55; white-space: nowrap;
  }
  .chip.key { opacity: .95; font-weight: 600; }
  .side-head {
    font-size: .68rem; text-transform: uppercase; letter-spacing: .1em;
    opacity: .5; margin: 1.1rem 0 .3rem;
  }
  .sample-head {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
    opacity: .45; margin-bottom: .35rem; display: flex; justify-content: space-between;
  }
  .sample {
    font-size: .95rem; line-height: 1.9; white-space: pre-wrap;
    word-break: break-word;
  }
  .sample .prompt { opacity: .45; }
  .caret {
    display: inline-block; width: .5em; opacity: .5;
    animation: blink 1s steps(2, start) infinite;
  }
  @keyframes blink { to { visibility: hidden; } }

  /* Per-token routing overlay. Background weight = adaptive layers taken;
     underline = the token queried the memory bank. Two orthogonal channels, so a
     token that did both is unambiguous. */
  .tok { border-radius: .2rem; padding: .06rem 0; }
  .tok.l1 { background: rgba(128,128,128,.10);
            background: color-mix(in srgb, currentColor 9%, transparent); }
  .tok.l2 { background: rgba(128,128,128,.17);
            background: color-mix(in srgb, currentColor 15%, transparent); }
  .tok.l3 { background: rgba(128,128,128,.24);
            background: color-mix(in srgb, currentColor 22%, transparent); }
  .tok.l4 { background: rgba(128,128,128,.32);
            background: color-mix(in srgb, currentColor 30%, transparent); }
  .tok.mem {
    text-decoration: underline; text-decoration-thickness: 2px;
    text-underline-offset: 3px; text-decoration-color: #8b5cf6;
    text-decoration-color: color-mix(in srgb, currentColor 25%, #8b5cf6);
  }
  .legend {
    display: flex; flex-wrap: wrap; gap: .9rem; align-items: center;
    font-size: .72rem; opacity: .55; margin: .2rem 0 1rem;
  }
  .legend .sw {
    display: inline-block; width: .8rem; height: .8rem; border-radius: .18rem;
    vertical-align: -.1rem; margin-right: .25rem;
    /* outlined so the zero-layer swatch, which has no fill, is still a swatch */
    box-shadow: inset 0 0 0 1px rgba(128,128,128,.4);
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="loading checkpoint...", max_entries=MAX_MODELS)
def load(ckpt_path, device):
    """Load one checkpoint, cached so a rerun does not re-read it off disk.

    `max_entries` bounds how many models can be resident at once. Without it, every
    checkpoint selected during a session stays in VRAM forever -- switching between
    six of them would OOM a 4GB card even though only one is on screen.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = AMT(cfg).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    return model, cfg, ck


@dataclass
class Model:
    """A loaded checkpoint and the shape facts the UI keeps asking it for."""
    path: str
    model: object
    cfg: object
    ck: dict
    routers: list

    @property
    def label(self):
        return label_for(self.path)

    @property
    def variant(self):
        return (self.ck.get("args", {}).get("variant")
                or run_variant(self.path) or "?")

    @property
    def n_adaptive(self):
        return len(self.cfg.adaptive_layers)

    @property
    def always_on(self):
        return self.cfg.n_layer - self.n_adaptive

    @property
    def stale(self):
        """Routers still sitting at the init threshold, so routing nothing away."""
        return [r.name for r in self.routers if float(r.threshold) == 0.0]

    @property
    def ppl(self):
        val = self.ck.get("val_loss")
        return math.exp(min(val, 20)) if val is not None else None

    @property
    def can_overlay(self):
        return bool(self.n_adaptive or self.cfg.use_memory)


def label_for(path):
    return f"{os.path.basename(os.path.dirname(path))} · {os.path.basename(path)}"


def bundle(path, device):
    """Wrap a cached checkpoint. Cheap -- the load underneath is what is cached."""
    model, cfg, ck = load(path, device)
    return Model(path=path, model=model, cfg=cfg, ck=ck, routers=routers_of(model))


@st.cache_resource
def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def find_checkpoints(root="runs"):
    """best.pt files first, then periodic ones, each newest-first.

    The best checkpoint is almost always the one worth looking at, so it should not
    be buried below whichever step a run happened to stop at.
    """
    def by_mtime(paths):
        return sorted(paths, key=os.path.getmtime, reverse=True)
    return (by_mtime(glob.glob(os.path.join(root, "*", "best.pt")))
            + by_mtime(glob.glob(os.path.join(root, "*", "ckpt_*.pt"))))


@st.cache_data(show_spinner=False)
def run_variant(ckpt_path):
    """The variant name for a checkpoint, from its run's config.json.

    Read from the sidecar rather than the checkpoint: labelling the picker must not
    cost a torch.load per entry, which on a directory of 200MB checkpoints would make
    the page unusable before anything is even selected.
    """
    meta = os.path.join(os.path.dirname(ckpt_path), "config.json")
    try:
        with open(meta) as f:
            return json.load(f).get("args", {}).get("variant", "")
    except (OSError, ValueError):
        return ""


def routers_of(model):
    return [m for m in model.modules() if isinstance(m, TopKTokenRouter)]


def router_order(name):
    """Sort key putting the memory router first and depth routers in layer order.

    Plain alphabetical sorting puts depth_router_10 before depth_router_4, which
    silently misreports which layer a routing rate belongs to.
    """
    if name == "mem_router":
        return (0, 0)
    tail = name.rsplit("_", 1)[-1]
    return (1, int(tail) if tail.isdigit() else 0)


def calibrate(model, cfg, data_dir, device):
    """Set every router's causal threshold from held-out data.

    Returns the per-router thresholds -- not rates. The threshold is the quantile of
    the aux predictor that *produces* the trained rate; reporting it as a rate makes
    a correctly calibrated router look badly off target.
    """
    loader = DocSegmentLoader(data_dir, 1, cfg.block_size, split="val", shuffle=False)
    amp, _, _ = select_precision(device, verbose=False)
    bank = model.make_bank(1, device, dtype=amp) if cfg.use_memory else None
    batches = [loader.next_batch()[0] for _ in range(3)]
    return model.calibrate_routers(batches, device=device, bank=bank)


def run_calibration(models, device):
    """Calibrate every model that has routers, stash the outcome, rerun.

    The rerun is what makes the header chips, the banner and the sidebar agree: they
    are rendered at different points in the script, so without it the page would show
    a stale "uncalibrated" above a fresh set of thresholds.

    One model failing does not abort the rest -- with several selected, a bad data
    directory for one is no reason to leave the others uncalibrated.
    """
    data_dir = st.session_state.get("data_dir", "data/fineweb_edu_docs")
    notes = []
    for m in models:
        if not m.routers:
            continue
        try:
            thresholds = calibrate(m.model, m.cfg, data_dir, device)
            notes.append(f"{m.label} · " + " · ".join(
                f"{k} {float(v):.3f}" for k, v in thresholds.items()))
        except Exception as exc:                          # noqa: BLE001
            notes.append(f"{m.label} failed: {exc}")
    st.session_state.calibration = notes
    st.rerun()


# ---------------------------------------------------------------------------
# Per-token telemetry
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One display unit of generated text and the routing that produced it.

    `depth` is the number of adaptive layers the token passed through and `mem`
    whether it queried the bank -- the decisions made while processing the context
    whose last position emitted this token, i.e. the compute that produced it.
    """
    text: str
    depth: float
    mem: bool
    conf: float


class Trace:
    """Records each router's decision for the token currently being generated.

    A forward hook per router, plus one on the memory block for the retrieval
    confidence -- `conf` is returned from `MemoryBlock.forward` and never passes
    through a router, so it is not reachable from the router hooks.

    Only position -1 is read. Earlier positions are re-routed on every step (there is
    no KV cache), so their decisions are recomputed context after context and belong
    to tokens already emitted.
    """

    def __init__(self, model):
        self.routers = routers_of(model)
        self.mem_blocks = [m for m in model.modules() if isinstance(m, MemoryBlock)]
        self.step = {}
        # Per-router running totals, kept alongside the per-token view because the
        # overlay records how many adaptive layers a token took, not which ones --
        # a stack that routes every token through layer 4 and none through layer 10
        # is indistinguishable from an even split once summed.
        self.counts = {}
        self.n_steps = 0
        self._handles = []

    def __enter__(self):
        self._handles = [r.register_forward_hook(self._on_router) for r in self.routers]
        self._handles += [m.register_forward_hook(self._on_memory) for m in self.mem_blocks]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _on_router(self, mod, inp, out):
        self.step[mod.name] = float(out["mask"][:, -1].float().mean())

    def _on_memory(self, mod, inp, out):
        # MemoryBlock.forward -> (x, conf, route_out, kv)
        self.step["conf"] = float(out[1][:, -1].float().mean())

    def read(self, cfg):
        """(depth, mem, conf) for the token just generated, then reset.

        With no memory router every token retrieves (route_memory=False, baseline B2),
        so the absence of a decision is itself the decision.
        """
        depth = sum(v for k, v in self.step.items() if k.startswith("depth_router_"))
        if "mem_router" in self.step:
            mem = self.step["mem_router"] > 0.5
        else:
            mem = bool(cfg.use_memory)
        conf = self.step.get("conf", 0.0)

        for name, v in self.step.items():
            if name != "conf":
                self.counts[name] = self.counts.get(name, 0.0) + v
        self.n_steps += 1

        self.step = {}
        return depth, mem, conf

    def rates(self):
        """Realised selection rate per router, over every token generated so far."""
        n = max(self.n_steps, 1)
        return {name: total / n for name, total in self.counts.items()}


def ends_sentence(text):
    """True when the text stops on something that looks like a sentence boundary.

    A heuristic, and an honest one about it: "Dr." and "e.g." read as endings here.
    Closing quotes and brackets are stepped over first so `... work."` counts, and a
    trailing newline counts on its own because a paragraph break ends a sentence
    whether or not the model punctuated it.
    """
    if text.endswith("\n"):
        return True
    t = text.rstrip()
    while t and t[-1] in '"\')]}”’':
        t = t[:-1]
    return bool(t) and t[-1] in ".!?…"


class Detok:
    """Turns sampled ids into display chunks with their routing attached.

    Decoding one BPE token at a time splits a multi-byte character into replacement
    chars, which looks like a model defect and is not one. Bytes are therefore held
    until they form valid UTF-8 and the held tokens are emitted as one chunk carrying
    their mean depth -- the alternative, decoding the whole continuation each step and
    diffing the strings, loses the token boundaries the routing overlay needs and
    strands the replacement char in the output once it has been shown.
    """

    MAX_HELD = 4      # longest UTF-8 sequence; beyond this the bytes are truly invalid

    def __init__(self, enc):
        self.enc = enc
        self.buf = b""
        self.held = []
        self.dropped = 0
        self.text = ""      # everything emitted so far, for the sentence-end check

    def push(self, token_id, depth, mem, conf):
        if token_id >= GPT2_VOCAB:
            self.dropped += 1        # padded vocab: trainable but not decodable
            return None
        self.buf += self.enc.decode_single_token_bytes(token_id)
        self.held.append((depth, mem, conf))
        try:
            return self._emit(self.buf.decode("utf-8"))
        except UnicodeDecodeError:
            if len(self.held) < self.MAX_HELD:
                return None
            return self._emit(self.buf.decode("utf-8", errors="replace"))

    def flush(self):
        """Emit whatever is still held, as replacement chars if it never completed."""
        if not self.held:
            return None
        return self._emit(self.buf.decode("utf-8", errors="replace"))

    def _emit(self, text):
        n = len(self.held)
        chunk = Chunk(text=text,
                      depth=sum(d for d, _, _ in self.held) / n,
                      mem=any(m for _, m, _ in self.held),
                      conf=max(c for _, _, c in self.held))
        self.buf, self.held = b"", []
        self.text += text
        return chunk


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def warm_bank(model, bank, ids, block_size, device):
    """Fill the bank segment by segment: forward, then write. Never the reverse."""
    bank.clear()
    written = 0
    with torch.no_grad():
        for s in range(0, len(ids), block_size):
            seg = ids[s:s + block_size]
            if len(seg) < 8:
                break
            x = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0)
            _, _, _, kv = model(x, bank=bank)
            model.write_memory(bank, kv)
            written += len(seg)
    return written


def stream(model, cfg, enc, prompt, n_tokens, temperature, top_k, seed, bank, device,
           trace, detok, finish=True, overrun=OVERRUN):
    """Yield (chunk or None, tokens sampled so far) as generation proceeds.

    None means the token's bytes are still incomplete or fell in the padded vocab.
    The count is yielded regardless so the progress bar tracks sampling rather than
    display.

    `n_tokens` is a budget, not a target. A fixed number of steps stops wherever it
    lands, which is mid-sentence far more often than not; with `finish` set the loop
    keeps sampling for up to `overrun` more tokens until the text reaches a sentence
    boundary. It is capped rather than open-ended because an undertrained model can
    run for hundreds of tokens without ever punctuating, and a playground that
    silently generates ten times what was asked for is worse than a clipped sentence.

    Sampling <|endoftext|> always stops, `finish` or not. Documents are stored
    EOT-prefixed (amt/data/prepare.py), so the model has learnt to emit it as a
    document break -- and it decodes to the literal string "<|endoftext|>", which
    would otherwise be rendered into the middle of a sample as if it were text.
    """
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    gen = torch.Generator(device=device).manual_seed(seed)
    produced = 0

    with trace:
        for _ in range(n_tokens + (overrun if finish else 0)):
            idx = model.generate(idx, 1, temperature=temperature, top_k=top_k,
                                 bank=bank, generator=gen)
            token = int(idx[0, -1])
            # Read the routing before the EOT test either way: it leaves the trace
            # empty for the next token, and the decision was really made.
            decision = trace.read(cfg)
            if token == EOT:
                break
            chunk = detok.push(token, *decision)
            produced += 1
            yield chunk, produced
            if produced >= n_tokens and (not finish or ends_sentence(detok.text)):
                break
    tail = detok.flush()
    if tail is not None:
        yield tail, produced


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def token_html(chunk, cfg, n_adaptive, always_on):
    text = html.escape(chunk.text)
    if not text:
        return ""
    if n_adaptive:
        level = round(RAMP * chunk.depth / n_adaptive)
        tip = f"{always_on + chunk.depth:.0f}/{cfg.n_layer} layers"
    else:
        level = 0
        tip = f"{cfg.n_layer} layers (dense)"
    if chunk.mem:
        tip += f" · retrieval {chunk.conf:.2f}"
    return f'<span class="tok l{level}{" mem" if chunk.mem else ""}" title="{tip}">{text}</span>'


def body_html(chunks, cfg, n_adaptive, always_on, overlay):
    if not overlay:
        return html.escape("".join(c.text for c in chunks))
    return "".join(token_html(c, cfg, n_adaptive, always_on) for c in chunks)


def render(slot, prompt, body, done=False):
    caret = "" if done else '<span class="caret">▍</span>'
    slot.markdown(
        f'<div class="sample"><span class="prompt">{html.escape(prompt)}</span>'
        f'{body}{caret}</div>', unsafe_allow_html=True)


def legend(models):
    """The overlay key, or "" when no selected model has anything to overlay.

    The swatches are labelled with absolute layer counts for a single model, because
    that is the number worth reading. Across several models the same swatch means a
    different layer count in each column, so the labels are dropped rather than
    printed wrong -- the ramp still reads as "more compute", which is the comparison
    being made.
    """
    depth_models = [m for m in models if m.n_adaptive]
    has_memory = any(m.cfg.use_memory for m in models)
    if not depth_models and not has_memory:
        return ""

    depth = ""
    if len(models) == 1 and depth_models:
        m = models[0]
        swatches = "".join(
            f'<span><span class="sw tok l{i}"></span>'
            f'{m.always_on + round(i * m.n_adaptive / RAMP)}</span>'
            for i in range(RAMP + 1))
        depth = f'<span>layers per token (of {m.cfg.n_layer}):</span>{swatches}'
    elif depth_models:
        swatches = "".join(f'<span class="sw tok l{i}"></span>' for i in range(RAMP + 1))
        depth = f'<span>fewer → more adaptive layers:</span><span>{swatches}</span>'

    mem = ('<span><span class="tok mem">underline</span> queried memory</span>'
           if has_memory else "")
    return f'<div class="legend">{depth}{mem}</div>'


def model_head(m, notes=()):
    """Column header: which checkpoint this is and what it cost to train."""
    chips = [f'<span class="chip key">{html.escape(m.variant)}</span>']
    if m.ppl is not None:
        chips.append(f'<span class="chip key">ppl {m.ppl:.1f}</span>')
    chips += [
        f'<span class="chip">step {m.ck.get("step", 0):,}</span>',
        f'<span class="chip">{FlopModel(m.cfg).breakdown().total/1e6:.1f} MFLOP/tok</span>',
    ]
    if m.routers:
        chips.append(f'<span class="chip{" key" if m.stale else ""}">'
                     f'{"uncalibrated" if m.stale else "calibrated"}</span>')
    return (f'<div class="model-name">{html.escape(m.label)}</div>'
            f'<div class="chips">{"".join(chips)}</div>')


def realised(m, data):
    """What this model's generation actually cost, from the recorded decisions."""
    chunks = [c for s in data["samples"] for c in s["chunks"]]
    n = len(chunks) or 1
    depth_rate = (sum(c.depth for c in chunks) / n / m.n_adaptive
                  if m.n_adaptive else 0.0)
    mem_rate = sum(1 for c in chunks if c.mem) / n
    fm = FlopModel(m.cfg)
    flops = fm.breakdown(depth_capacity=depth_rate, mem_capacity=mem_rate).total
    return {
        "layers": m.always_on + depth_rate * m.n_adaptive,
        "trained_layers": m.always_on + m.cfg.effective_depth_capacity * m.n_adaptive,
        "depth_rate": depth_rate,
        "mem_rate": mem_rate,
        "mflops": flops / 1e6,
        "of_dense": flops / fm.dense_breakdown().total,
        "tok_s": data["tokens"] / max(data["secs"], 1e-6),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpts = find_checkpoints()

if not ckpts:
    st.title("AMT playground")
    st.error("No checkpoints under `runs/`. Train something, or drop a `best.pt` "
             "into `runs/<name>/`.")
    st.stop()


def side_head(label):
    st.markdown(f'<div class="side-head">{label}</div>', unsafe_allow_html=True)


with st.sidebar:
    side_head("checkpoints")
    selected = st.multiselect(
        "checkpoints", ckpts, default=ckpts[:1], label_visibility="collapsed",
        format_func=label_for, max_selections=MAX_MODELS,
        help="each checkpoint gets its own column and sees the same prompt, the same "
             "seeds and the same memory context")

    side_head("sampling")
    n_samples = st.slider("samples", 1, 6, 3)
    n_tokens = st.slider("tokens each", 16, 512, 160, 16)
    temperature = st.slider("temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("top-k", 1, 200, 50, 1)
    seed = st.number_input(
        "seed", value=1337, step=1,
        help="sample 1 uses this seed; the rest are drawn at random, and each card "
             "shows the seed it used so a good one can be pinned here")
    finish = st.toggle(
        "finish the sentence", value=True,
        help=f"treat 'tokens each' as a minimum and keep sampling up to {OVERRUN} "
             "more tokens to land on a sentence boundary")

st.title("AMT playground")

if not selected:
    st.info("Pick one or more checkpoints in the sidebar. Several run the same prompt "
            "through each model, side by side.")
    st.stop()

models = [bundle(p, device) for p in selected]

# The page widens with the column count instead of being wide all the time: prose at
# 52rem is the readable measure, and a single model should not be stretched across a
# monitor just because comparing four of them needs the room.
st.markdown(f"<style>.block-container {{ max-width: {WIDTHS.get(len(models), 104)}rem; }}"
            "</style>", unsafe_allow_html=True)

with st.sidebar:
    if any(m.routers for m in models):
        side_head("routing")
        # Read back through session_state (key="data_dir") so the banner button below
        # can calibrate with the same directory without a second widget.
        st.text_input("data dir", "data/fineweb_edu_docs", key="data_dir",
                      help="held-out batches for router calibration")
        if st.button("calibrate routers", use_container_width=True):
            run_calibration(models, device)
        for note in st.session_state.get("calibration") or []:
            (st.error if "failed" in note else st.caption)(note)

st.markdown(f'<div class="chips"><span class="chip">{device}</span></div>',
            unsafe_allow_html=True)

# An uncalibrated threshold means the causal router selects every token, so the model
# generating is dense and is not the model that was trained. That is not a sidebar
# detail -- it invalidates every column it applies to.
uncal = [m for m in models if m.stale]
if uncal:
    warn, act = st.columns([4, 1], vertical_alignment="center")
    warn.warning("uncalibrated: " + ", ".join(m.label for m in uncal)
                 + " — causal routing selects every token there, so those columns "
                   "come from a dense model, not the trained one.")
    if act.button("calibrate", type="primary", use_container_width=True):
        run_calibration(uncal, device)

# -- inputs ----------------------------------------------------------------

context = ""
if any(m.cfg.use_memory for m in models):
    with st.expander("memory context — fills the bank before generating"):
        st.caption(
            "`AMT.generate` reads the memory bank but never writes to it. Without "
            "context here the bank stays empty, retrieval contributes nothing, and "
            "you are looking at a depth-routed model rather than the joint one. Each "
            "memory model gets its own bank, warmed from this same text.")
        context = st.text_area("context", height=130, label_visibility="collapsed")

prompt = st.text_area("prompt", "The most important idea in this paper is", height=80)
go = st.button("generate", type="primary", use_container_width=True)
if len(models) > 1:
    st.caption(
        f"{len(models)} models, run one after another on the same prompt and seeds. "
        "Sequential is not a limitation to fix: there is one GPU, so interleaving them "
        "would not finish sooner and would hold every model's activations at once.")

# -- run -------------------------------------------------------------------

if go:
    enc = tokenizer()
    amp_dtype, _, _ = select_precision(device, verbose=False)
    budget = int(n_tokens)
    # Sample 1 is the reproducible one; the rest are drawn fresh on every press.
    # Consecutive seeds (an old seed + i) are not independent draws in any useful
    # sense -- they are a single arbitrary slice of seed space, so several samples of
    # the same prompt could agree simply for having been chosen next to each other.
    # Every card shows its seed, so a sample worth keeping can be pinned in the box.
    # The same list is used for every model: same seed, same prompt, different model
    # is the only comparison that isolates the model.
    seeds = [int(seed)] + [random.randrange(2**31 - 1) for _ in range(n_samples - 1)]

    st.divider()
    for col, m in zip(st.columns(len(models)), models):
        col.markdown(model_head(m), unsafe_allow_html=True)
    st.markdown(legend(models), unsafe_allow_html=True)

    # Build the whole grid up front so the layout does not jump as it fills in. Rows
    # are samples and columns are models, which puts the same seed side by side.
    grid = []
    for r in range(n_samples):
        row = []
        for col, m in zip(st.columns(len(models)), models):
            with col, st.container(border=True):
                st.markdown(f'<div class="sample-head"><span>sample {r+1}</span>'
                            f'<span>seed {seeds[r]}</span></div>',
                            unsafe_allow_html=True)
                row.append(st.empty())
                render(row[-1], prompt, "", done=True)
        grid.append(row)

    progress = st.progress(0.0)
    total = max(1, len(models) * n_samples * budget)
    out = {}

    for ci, m in enumerate(models):
        notes = []
        bank = (m.model.make_bank(1, device, dtype=amp_dtype)
                if m.cfg.use_memory else None)
        if bank is not None and context.strip():
            n = warm_bank(m.model, bank, enc.encode(context), m.cfg.block_size, device)
            notes.append(f"bank warmed with {n} tokens · fill "
                         f"{float(bank.fill.float().mean()):.0f}/{bank.capacity}")
        elif bank is not None:
            notes.append("bank empty — retrieval contributes nothing")

        trace = Trace(m.model)
        samples = []
        started = time.perf_counter()

        with torch.no_grad():
            for ri in range(n_samples):
                slot = grid[ri][ci]
                detok = Detok(enc)
                chunks, parts, produced = [], [], 0
                for chunk, produced in stream(m.model, m.cfg, enc, prompt, budget,
                                              temperature, int(top_k), seeds[ri], bank,
                                              device, trace, detok, finish=finish):
                    if chunk is not None:
                        chunks.append(chunk)
                        parts.append(token_html(chunk, m.cfg, m.n_adaptive, m.always_on))
                        render(slot, prompt, "".join(parts))
                    # Clamped: finishing a sentence runs past the budget the bar is
                    # sized on, and a fraction over 1.0 is an error, not a full bar.
                    done = (ci * n_samples + ri) * budget + produced
                    progress.progress(min(1.0, done / total),
                                      text=f"{m.label} · sample {ri+1}/{n_samples}")
                render(slot, prompt, "".join(parts), done=True)
                samples.append({"seed": seeds[ri], "chunks": chunks,
                                "dropped": detok.dropped, "tokens": produced,
                                "ended": ends_sentence(detok.text)})

        out[m.path] = {"samples": samples, "notes": notes, "rates": trace.rates(),
                       "secs": time.perf_counter() - started,
                       "tokens": sum(s["tokens"] for s in samples)}
        # Drop this model's bank before the next one allocates its own. The bank is
        # small next to the weights, but on a 4GB card holding four of them for no
        # reason is the difference between comparing four models and OOMing on the
        # fourth.
        del bank
        if device == "cuda":
            torch.cuda.empty_cache()
    progress.empty()

    st.session_state.result = {"prompt": prompt, "budget": budget, "models": out}
    st.rerun()

# -- results ---------------------------------------------------------------

result = st.session_state.get("result") or {}
# Only columns whose checkpoint is still selected: routing decisions belong to the
# model that made them, so a deselected model's samples go rather than mislabel.
shown = [m for m in models if m.path in result.get("models", {})]

if shown:
    st.divider()

    # A dense baseline routes nothing and retrieves nothing, so there is no overlay to
    # draw and the control would be a switch that does nothing.
    overlay = any(m.can_overlay for m in shown) and st.toggle(
        "show routing", value=True,
        help="background weight = adaptive layers taken · "
             "underline = queried the memory bank")

    for col, m in zip(st.columns(len(shown)), shown):
        col.markdown(model_head(m), unsafe_allow_html=True)
        for note in result["models"][m.path]["notes"]:
            col.caption(note)

    if overlay:
        st.markdown(legend(shown), unsafe_allow_html=True)

    n_rows = max(len(result["models"][m.path]["samples"]) for m in shown)
    for ri in range(n_rows):
        for col, m in zip(st.columns(len(shown)), shown):
            rows = result["models"][m.path]["samples"]
            if ri >= len(rows):
                continue
            sample = rows[ri]
            with col, st.container(border=True):
                st.markdown(f'<div class="sample-head"><span>sample {ri+1}</span>'
                            f'<span>seed {sample["seed"]}</span></div>',
                            unsafe_allow_html=True)
                render(st, result["prompt"],
                       body_html(sample["chunks"], m.cfg, m.n_adaptive, m.always_on,
                                 overlay),
                       done=True)

    # -- realised cost -----------------------------------------------------

    st.divider()
    st.caption("routing during generation — realised, not the trained capacity")
    stats = {m.path: realised(m, result["models"][m.path]) for m in shown}

    if len(shown) == 1:
        m, s = shown[0], stats[shown[0].path]
        cards = [("layers / token", f"{s['layers']:.2f}",
                  f"{s['layers'] - s['trained_layers']:+.2f} vs trained")]
        if m.cfg.use_memory:    # 0.00 against a target of 0.00 says nothing
            cards.append(("memory rate", f"{s['mem_rate']:.2f}",
                          f"{s['mem_rate'] - m.cfg.effective_mem_capacity:+.2f} "
                          "vs trained"))
        cards += [("MFLOP / token", f"{s['mflops']:.1f}", f"{s['of_dense']:.0%} of dense"),
                  ("tokens / s", f"{s['tok_s']:.1f}", None)]
        for col, (label, value, delta) in zip(st.columns(len(cards)), cards):
            col.metric(label, value, delta, delta_color="off")
    else:
        # ppl comes from the checkpoint's own val loss, not from anything measured
        # here: a handful of sampled continuations says nothing about quality, and a
        # column that generates prettier text at higher perplexity is exactly the
        # trade this table exists to make visible.
        table = ["| model | variant | ppl | layers/tok | mem rate | MFLOP/tok | "
                 "vs dense | tok/s |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for m in shown:
            s = stats[m.path]
            ppl = f"{m.ppl:.1f}" if m.ppl is not None else "—"
            mem = f"{s['mem_rate']:.2f}" if m.cfg.use_memory else "—"
            table.append(f"| {m.label} | {m.variant} | {ppl} | {s['layers']:.2f} "
                         f"| {mem} | {s['mflops']:.1f} | {s['of_dense']:.0%} "
                         f"| {s['tok_s']:.1f} |")
        st.markdown("\n".join(table))
        st.caption(
            "ppl is each checkpoint's own validation loss, carried in the file — not "
            "measured from these samples. A cheaper model that costs perplexity is a "
            "trade, not a loss; read the two columns together, and see "
            "`scripts/compare.py table` for the same comparison over whole runs.")

    for m in shown:
        data = result["models"][m.path]
        prefix = f"{m.label}: " if len(shown) > 1 else ""
        clipped = [str(i + 1) for i, s in enumerate(data["samples"]) if not s["ended"]]
        if clipped:
            st.caption(
                f"{prefix}sample {', '.join(clipped)} still ends mid-sentence: the "
                f"model ran the whole {result['budget']}-token budget (plus any "
                "overrun) without punctuating. Raise `tokens each`, or read it as the "
                "model's own behaviour — an undertrained one often never closes a "
                "sentence.")
        dropped = sum(s["dropped"] for s in data["samples"])
        if dropped:
            st.caption(
                f"{prefix}{dropped} sampled ids fell in the padded vocab "
                f"({GPT2_VOCAB}–{m.cfg.vocab_size - 1}) and were dropped on decode. "
                "Expected early in training; persistent means the head has mass on "
                "tokens that never occur.")

    for m in shown:
        rates = result["models"][m.path].get("rates") or {}
        if not rates:
            continue
        title = ("per-router rates" if len(shown) == 1
                 else f"per-router rates — {m.label}")
        with st.expander(title):
            for name, got in sorted(rates.items(), key=lambda kv: router_order(kv[0])):
                target = (m.cfg.mem_capacity if name == "mem_router"
                          else m.cfg.depth_capacity)
                label = (name.replace("depth_router_", "depth ")
                             .replace("mem_router", "memory"))
                st.progress(min(got, 1.0),
                            text=f"{label} — {got:.3f} realised · {target:.2f} trained")
            st.caption(
                "One layer routing every token while another routes none sums to the "
                "same layers/token as an even split, so the per-layer view is the one "
                "that shows a collapsed stack.")

    st.caption(
        "A realised rate far from the trained capacity means the causal threshold is "
        "not reproducing the top-k selection used in training — check `agree/*` in the "
        "run log before reading anything into the text. MFLOP/token is the analytic "
        "matmul model at the realised capacities and excludes retrieval's bandwidth "
        "cost (amt/model/flops.py).")
