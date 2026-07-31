"""Streamlit UI for poking at a trained AMT checkpoint.

    streamlit run scripts/app.py

The same three traps as scripts/infer.py apply here, and are handled the same way:
router thresholds need calibrating before causal routing means anything, generation
never writes to the memory bank so it must be pre-warmed to observe the memory half,
and the vocab is padded past what GPT-2 BPE can decode.

Sampling deliberately calls `AMT.generate` one token at a time rather than
reimplementing the loop. There is no KV cache in this model -- `generate` re-runs the
forward over the whole context every step regardless -- so a 1-token call costs
exactly what one iteration of its own loop costs, and the sampling logic stays in one
place. The `torch.Generator` is created once per sample and threaded through, or every
token would be drawn from the same seed.
"""

import glob
import html
import os
import sys

import streamlit as st
import torch

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT, FlopModel  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import select_precision  # noqa: E402

GPT2_VOCAB = 50257

st.set_page_config(page_title="AMT playground", page_icon="🧠", layout="centered")

# Styling stays theme-agnostic: opacity and borders rather than fixed colours, so it
# reads correctly whether the viewer is in Streamlit's light or dark theme.
st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; max-width: 52rem; }
  .chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: -.5rem 0 1.4rem; }
  .chip {
    font-size: .74rem; padding: .18rem .55rem; border-radius: 1rem;
    border: 1px solid currentColor; opacity: .55; white-space: nowrap;
  }
  .chip.key { opacity: .95; font-weight: 600; }
  .sample-head {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
    opacity: .45; margin-bottom: .35rem;
  }
  .sample {
    font-size: .95rem; line-height: 1.75; white-space: pre-wrap;
    word-break: break-word;
  }
  .sample .prompt { opacity: .45; }
  .caret {
    display: inline-block; width: .5em; opacity: .5;
    animation: blink 1s steps(2, start) infinite;
  }
  @keyframes blink { to { visibility: hidden; } }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="loading checkpoint...")
def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = AMT(cfg).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    return model, cfg, ck


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


def routers_of(model):
    return [m for m in model.modules() if isinstance(m, TopKTokenRouter)]


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


def stream(model, enc, prompt, n_tokens, temperature, top_k, seed, bank, device,
           telemetry):
    """Yield decoded text deltas as tokens are sampled."""
    handles = []
    if telemetry is not None:
        def hook(mod, inp, out):
            telemetry.setdefault(mod.name, []).append(
                float(out["mask"][:, -1].float().mean()))
        handles = [r.register_forward_hook(hook) for r in routers_of(model)]

    try:
        ids = enc.encode(prompt)
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        gen = torch.Generator(device=device).manual_seed(seed)

        emitted, shown = [], ""
        for _ in range(n_tokens):
            idx = model.generate(idx, 1, temperature=temperature, top_k=top_k,
                                 bank=bank, generator=gen)
            emitted.append(int(idx[0, -1]))
            # Decode the whole continuation each step and yield the delta: decoding
            # one BPE token at a time splits multi-byte characters into replacement
            # chars, which looks like a model defect and is not one.
            text = enc.decode([t for t in emitted if t < GPT2_VOCAB])
            yield text[len(shown):], sum(1 for t in emitted if t >= GPT2_VOCAB)
            shown = text
    finally:
        for h in handles:
            h.remove()


def render(slot, prompt, body, done=False):
    caret = "" if done else '<span class="caret">▍</span>'
    slot.markdown(
        f'<div class="sample"><span class="prompt">{html.escape(prompt)}</span>'
        f'{html.escape(body)}{caret}</div>', unsafe_allow_html=True)


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

with st.sidebar:
    ckpt_path = st.selectbox(
        "checkpoint", ckpts,
        format_func=lambda p: f"{os.path.basename(os.path.dirname(p))} · "
                              f"{os.path.basename(p)}")
    model, cfg, ck = load(ckpt_path, device)
    routers = routers_of(model)

    st.divider()
    n_samples = st.slider("samples", 1, 6, 3)
    n_tokens = st.slider("tokens each", 16, 512, 160, 16)
    temperature = st.slider("temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("top-k", 1, 200, 50, 1)
    seed = st.number_input("seed", value=1337, step=1,
                           help="sample i uses seed + i")

    if routers:
        st.divider()
        stale = [r.name for r in routers if float(r.threshold) == 0.0]
        if stale:
            st.warning(f"{len(stale)} router(s) uncalibrated — causal routing "
                       "selects every token until you calibrate.")
        data_dir = st.text_input("data dir", "data/fineweb_edu_docs",
                                 help="held-out batches for router calibration")
        if st.button("calibrate routers", use_container_width=True):
            try:
                loader = DocSegmentLoader(data_dir, 1, cfg.block_size,
                                          split="val", shuffle=False)
                amp, _, _ = select_precision(device, verbose=False)
                b = model.make_bank(1, device, dtype=amp) if cfg.use_memory else None
                rates = model.calibrate_routers(
                    [loader.next_batch()[0] for _ in range(3)], device=device, bank=b)
                st.success(" · ".join(f"{k} {float(v):.3f}" for k, v in rates.items()))
            except Exception as exc:                      # noqa: BLE001
                st.error(f"calibration failed: {exc}")

# -- header ----------------------------------------------------------------

st.title("AMT playground")

targs = ck.get("args", {})
val = ck.get("val_loss")
chips = [
    f'<span class="chip key">{targs.get("variant", "?")}</span>',
    f'<span class="chip">step {ck.get("step", 0):,}</span>',
    f'<span class="chip">{model.num_params()/1e6:.1f}M non-emb</span>',
    f'<span class="chip">{FlopModel(cfg).breakdown().total/1e6:.1f} MFLOP/tok</span>',
    f'<span class="chip">{device}</span>',
]
if val is not None:
    import math
    chips.insert(2, f'<span class="chip key">ppl {math.exp(min(val, 20)):.1f}</span>')
if cfg.route_depth:
    chips.append(f'<span class="chip">depth cap {cfg.depth_capacity}</span>')
if cfg.use_memory:
    chips.append(f'<span class="chip">mem cap {cfg.mem_capacity}</span>')
st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)

# -- inputs ----------------------------------------------------------------

context = ""
if cfg.use_memory:
    with st.expander("memory context — fills the bank before generating"):
        st.caption(
            "`AMT.generate` reads the memory bank but never writes to it. Without "
            "context here the bank stays empty, retrieval contributes nothing, and "
            "you are looking at a depth-routed model rather than the joint one.")
        context = st.text_area("context", height=130, label_visibility="collapsed")

prompt = st.text_area("prompt", "The most important idea in this paper is",
                      height=80)
go = st.button("generate", type="primary", use_container_width=True)

# -- run -------------------------------------------------------------------

if go:
    enc = tokenizer()
    amp_dtype, _, _ = select_precision(device, verbose=False)
    bank = model.make_bank(1, device, dtype=amp_dtype) if cfg.use_memory else None

    if bank is not None and context.strip():
        n = warm_bank(model, bank, enc.encode(context), cfg.block_size, device)
        st.caption(f"bank warmed with {n} tokens · fill "
                   f"{float(bank.fill.float().mean()):.0f}/{bank.capacity}")
    elif bank is not None:
        st.caption("bank empty — retrieval will contribute nothing")

    telemetry = {} if routers else None
    bad_total = 0
    progress = st.progress(0.0)

    # Build every card up front so the layout does not jump as samples fill in.
    slots = []
    for i in range(n_samples):
        with st.container(border=True):
            st.markdown(f'<div class="sample-head">sample {i+1} · seed '
                        f'{int(seed) + i}</div>', unsafe_allow_html=True)
            slots.append(st.empty())
    for slot in slots:
        render(slot, prompt, "", done=True)

    with torch.no_grad():
        for i, slot in enumerate(slots):
            body = ""
            for delta, bad in stream(model, enc, prompt, int(n_tokens), temperature,
                                     int(top_k), int(seed) + i, bank, device,
                                     telemetry):
                body += delta
                render(slot, prompt, body)
            render(slot, prompt, body, done=True)
            bad_total += bad
            progress.progress((i + 1) / n_samples)
    progress.empty()

    if bad_total:
        st.caption(
            f"{bad_total} sampled ids fell in the padded vocab "
            f"({GPT2_VOCAB}–{cfg.vocab_size - 1}) and were dropped on decode. "
            "Expected early in training; persistent means the head has mass on "
            "tokens that never occur.")

    if telemetry:
        st.divider()
        st.caption("routing during generation — realised rate vs trained capacity")
        items = sorted(telemetry.items())
        cols = st.columns(min(len(items), 4))
        for i, (name, vals) in enumerate(items):
            target = cfg.mem_capacity if name == "mem_router" else cfg.depth_capacity
            got = sum(vals) / len(vals)
            cols[i % len(cols)].metric(name.replace("depth_router_", "depth "),
                                       f"{got:.2f}", f"{got - target:+.2f}",
                                       delta_color="off")
        st.caption(
            "A realised rate far from the trained capacity means the causal "
            "threshold is not reproducing the top-k selection used in training — "
            "check `agree/*` in the run log before reading anything into the text.")
