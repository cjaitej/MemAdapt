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
place. The `torch.Generator` is created once and threaded through, or every token
would be drawn from the same seed.
"""

import glob
import os
import sys

import streamlit as st
import torch

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT, FlopModel  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import describe_device, select_precision  # noqa: E402

GPT2_VOCAB = 50257

st.set_page_config(page_title="AMT playground", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="loading checkpoint...")
def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = AMT(cfg).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    return model, cfg, ck.get("step", 0), ck.get("args", {})


@st.cache_resource
def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def find_checkpoints(root="runs"):
    """best.pt files first, then periodic ones, each newest-first.

    The best checkpoint is almost always the one worth looking at, so it should not
    be buried below whichever step a run happened to stop at.
    """
    by_mtime = lambda ps: sorted(ps, key=os.path.getmtime, reverse=True)  # noqa: E731
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
            yield text[len(shown):]
            shown = text
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpts = find_checkpoints()

st.title("AMT playground")

if not ckpts:
    st.error("No checkpoints found under `runs/`. Train something, or copy a "
             "`ckpt_*.pt` into `runs/<name>/`.")
    st.stop()

with st.sidebar:
    st.caption(describe_device())
    ckpt_path = st.selectbox("checkpoint", ckpts,
                             format_func=lambda p: os.path.join(
                                 os.path.basename(os.path.dirname(p)),
                                 os.path.basename(p)))
    model, cfg, step, targs = load(ckpt_path, device)
    routers = routers_of(model)

    st.markdown(
        f"**{targs.get('variant', '?')}** · step {step:,} · "
        f"{model.num_params()/1e6:.1f}M non-emb\n\n"
        f"`{FlopModel(cfg).breakdown().total/1e6:.1f}` MFLOP/tok · "
        f"depth_cap `{cfg.depth_capacity}` · mem_cap `{cfg.mem_capacity}`"
    )

    st.divider()
    n_tokens = st.slider("max new tokens", 16, 512, 200, 16)
    temperature = st.slider("temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("top-k", 1, 200, 50, 1)
    seed = st.number_input("seed", value=1337, step=1)

    if routers:
        st.divider()
        st.subheader("routing")
        stale = [r.name for r in routers if float(r.threshold) == 0.0]
        if stale:
            st.warning(
                f"{len(stale)} router(s) uncalibrated. Causal routing selects "
                "every token until you calibrate, so this is a dense model with "
                "extra steps.")
        data_dir = st.text_input("data dir (for calibration)",
                                 "data/fineweb_edu_docs")
        if st.button("calibrate routers", use_container_width=True):
            try:
                loader = DocSegmentLoader(data_dir, 1, cfg.block_size,
                                          split="val", shuffle=False)
                amp, _, _ = select_precision(device, verbose=False)
                b = (model.make_bank(1, device, dtype=amp) if cfg.use_memory
                     else None)
                rates = model.calibrate_routers(
                    [loader.next_batch()[0] for _ in range(3)], device=device,
                    bank=b)
                st.success(" · ".join(f"{k} {float(v):.3f}"
                                      for k, v in rates.items()))
            except Exception as exc:                      # noqa: BLE001
                st.error(f"calibration failed: {exc}")

context = ""
if cfg.use_memory:
    with st.expander("memory context — fills the bank before generating", expanded=False):
        st.caption(
            "`AMT.generate` reads the memory bank but never writes to it. Without "
            "context here the bank stays empty, retrieval contributes nothing, and "
            "you are looking at a depth-routed model rather than the joint one.")
        context = st.text_area("context passage", height=140,
                               label_visibility="collapsed")

prompt = st.text_area("prompt", "The most important idea in this paper is",
                      height=90)
go = st.button("generate", type="primary")

if go:
    enc = tokenizer()
    amp_dtype, _, _ = select_precision(device, verbose=False)
    bank = model.make_bank(1, device, dtype=amp_dtype) if cfg.use_memory else None

    if bank is not None and context.strip():
        n = warm_bank(model, bank, enc.encode(context), cfg.block_size, device)
        st.caption(f"bank warmed with {n} tokens "
                   f"(fill {float(bank.fill.float().mean()):.0f}/{bank.capacity})")
    elif bank is not None:
        st.caption("bank empty — retrieval will contribute nothing")

    telemetry = {} if routers else None
    with torch.no_grad():
        st.write_stream(stream(model, enc, prompt, int(n_tokens), temperature,
                               int(top_k), int(seed), bank, device, telemetry))

    if telemetry:
        st.divider()
        st.caption("routing during generation — realised rate vs trained capacity")
        cols = st.columns(min(len(telemetry), 4))
        for i, (name, vals) in enumerate(sorted(telemetry.items())):
            target = cfg.mem_capacity if name == "mem_router" else cfg.depth_capacity
            got = sum(vals) / len(vals)
            cols[i % len(cols)].metric(name, f"{got:.2f}", f"{got - target:+.2f}",
                                       delta_color="off")
        st.caption(
            "A realised rate far from the trained capacity means the causal "
            "threshold is not reproducing the top-k selection used in training — "
            "check `agree/*` in the run log before reading anything into the text.")
