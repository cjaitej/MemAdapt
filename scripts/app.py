"""Streamlit demo: see which tokens the model decided were easy.

    streamlit run scripts/app.py

What this shows that a table cannot
-----------------------------------
"6.2 layers per token" is a budget. The claim is about *allocation* -- that the model
spends its depth on the tokens that need it -- and an average cannot show allocation
either way. Per-token shading can: function words, punctuation and the tail of a
predictable word should come out pale, and content words at the start of a clause
should come out dark. If the shading looks like noise, the average was a budget the
router hit rather than a decision it made, and the `random` baseline in
`scripts/compare.py` will say so numerically.

Two views, and the difference matters
-------------------------------------
* **Analyse** runs one teacher-forced pass over text you paste, so every token's depth
  is the depth it would get while *reading*. This is the setting the perplexity
  numbers come from.
* **Generate** samples one token at a time, so each depth belongs to a token the model
  had not seen yet. Same routers, same decisions -- an exit decision reads one token's
  own hidden state and nothing else, so it needs no calibration and there is no
  train/generate gap to caveat.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import html

import streamlit as st
import tiktoken
import torch

from agpt.evaluate import load_checkpoint

# Sequential blue ramp, light -> dark, from the validated palette. Depth is a
# magnitude, so it gets a single-hue ramp; the categorical slots are for the four arms
# in the figures and are deliberately not reused here.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Below this step the background is light enough that dark ink reads better.
INK_FLIP = 6


def shade(depth, n_layer, min_depth):
    """Background and text colour for a token at this depth.

    Scaled between the model's minimum and maximum possible depth rather than 0 and
    n_layer: with an always-on prefix of 3, no token can ever be paler than step 3, and
    a ramp that reserves a quarter of its range for values that cannot occur throws
    away the contrast the figure needs.
    """
    span = max(n_layer - min_depth, 1)
    t = min(max((depth - min_depth) / span, 0.0), 1.0)
    i = round(t * (len(RAMP) - 1))
    return RAMP[i], ("#0b0b0b" if i < INK_FLIP else "#ffffff")


def render_tokens(pieces, depths, n_layer, min_depth):
    spans = []
    for piece, depth in zip(pieces, depths):
        bg, fg = shade(depth, n_layer, min_depth)
        text = html.escape(piece).replace("\n", "<br>").replace(" ", "&nbsp;")
        spans.append(
            f'<span title="{depth:.0f} layers" style="background:{bg};color:{fg};'
            f'padding:2px 1px;border-radius:3px;">{text}</span>')
    return (f'<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;'
            f'font-size:14px;line-height:2.1;">{"".join(spans)}</div>')


def legend(n_layer, min_depth):
    cells = []
    for d in range(min_depth, n_layer + 1):
        bg, fg = shade(d, n_layer, min_depth)
        cells.append(f'<span style="background:{bg};color:{fg};padding:3px 7px;'
                     f'border-radius:3px;font-size:12px;">{d}</span>')
    return ('<div style="display:flex;gap:3px;align-items:center;flex-wrap:wrap;">'
            '<span style="font-size:12px;color:#52514e;margin-right:6px;">'
            'layers computed</span>' + "".join(cells) + "</div>")


@st.cache_resource(show_spinner="loading checkpoint …")
def load(path, exit_mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    over = {"exit_mode": exit_mode} if exit_mode != "(as saved)" else {}
    model, cfg, ck = load_checkpoint(path, device, **over)
    return model, cfg, ck, device


@st.cache_resource
def tokenizer():
    return tiktoken.get_encoding("gpt2")


def decode_pieces(enc, ids):
    """One display string per token id, with undecodable padding ids marked.

    `vocab_size` is padded to a multiple of 128 for tensor-core alignment, so a trained
    model can sample an id GPT-2's BPE has no token for. Showing those rather than
    crashing on them makes it obvious when the head is putting mass on tokens that do
    not exist.
    """
    out = []
    for i in ids:
        try:
            out.append(enc.decode([int(i)]))
        except Exception:                                  # noqa: BLE001
            out.append("�")
    return out


st.set_page_config(page_title="AdaptiveGPT", layout="wide")
st.title("AdaptiveGPT — where the depth went")

with st.sidebar:
    st.header("model")
    ckpt = st.text_input("checkpoint", "runs/s3_joint/best.pt")
    exit_mode = st.selectbox(
        "routing rule", ["(as saved)", "adaptive", "dense", "fixed", "random"],
        help="The baselines are the same weights under a different rule -- exactly "
             "what scripts/compare.py evaluates.")
    st.caption("Switching the rule rebuilds the model from the same checkpoint, so "
               "any difference you see is the routing and nothing else.")

if not os.path.exists(ckpt):
    st.warning(f"No checkpoint at `{ckpt}`. Train one first:\n\n"
               "```\npython -m agpt.train --stage dense --run-name s1_dense\n```")
    st.stop()

try:
    model, cfg, ck, device = load(ckpt, exit_mode)
except Exception as e:                                     # noqa: BLE001
    st.error(f"could not load `{ckpt}`: {e}")
    st.stop()

enc = tokenizer()
c1, c2, c3, c4 = st.columns(4)
c1.metric("routing", cfg.exit_mode)
c2.metric("layers", cfg.n_layer)
c3.metric("always-on prefix", cfg.n_min_layers)
c4.metric("step", ck.get("step", "?"))

analyse, generate = st.tabs(["Analyse text", "Generate"])

with analyse:
    text = st.text_area(
        "text", height=160,
        value="The capital of France is Paris. It is a city that has been the "
              "capital since the tenth century, and it remains the largest "
              "settlement in the country by a considerable margin.")
    if st.button("Analyse", type="primary") and text.strip():
        ids = enc.encode_ordinary(text)[:cfg.block_size]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            depths = model.token_depths(x)[0].float().cpu().tolist()

        st.markdown(legend(cfg.n_layer, cfg.min_depth), unsafe_allow_html=True)
        st.markdown(render_tokens(decode_pieces(enc, ids), depths,
                                  cfg.n_layer, cfg.min_depth),
                    unsafe_allow_html=True)

        a, b = st.columns([1, 2])
        a.metric("mean depth", f"{sum(depths)/len(depths):.2f} / {cfg.n_layer}")
        counts = [0] * (cfg.n_layer + 1)
        for d in depths:
            counts[int(d)] += 1
        b.bar_chart({"tokens": counts}, height=180)

        deep = sorted(zip(depths, decode_pieces(enc, ids)), reverse=True)[:12]
        shallow = sorted(zip(depths, decode_pieces(enc, ids)))[:12]
        d1, d2 = st.columns(2)
        d1.caption("deepest tokens")
        d1.write(" · ".join(f"`{t.strip() or '␣'}`" for _, t in deep))
        d2.caption("shallowest tokens")
        d2.write(" · ".join(f"`{t.strip() or '␣'}`" for _, t in shallow))

with generate:
    prompt = st.text_input("prompt", "The capital of France is")
    n = st.slider("tokens", 8, 256, 64)
    temp = st.slider("temperature", 0.1, 1.5, 0.8)
    seed = st.number_input("seed", value=1337, step=1)

    if st.button("Generate", type="primary"):
        ids = enc.encode_ordinary(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        with st.spinner("sampling …"):
            with torch.no_grad():
                # One forward per token, so each reported depth belongs to exactly one
                # generated token. There is no KV cache in this model -- `generate`
                # re-runs the whole context every step regardless -- so this costs
                # nothing over batching the loop.
                out, depths = model.generate(x, int(n), temperature=temp,
                                             generator=gen, return_depth=True)
        new_ids = out[0, len(ids):].tolist()
        vals = [float(d) for d in depths]

        st.markdown(legend(cfg.n_layer, cfg.min_depth), unsafe_allow_html=True)
        st.markdown(f"**{html.escape(prompt)}**", unsafe_allow_html=True)
        st.markdown(render_tokens(decode_pieces(enc, new_ids), vals,
                                  cfg.n_layer, cfg.min_depth),
                    unsafe_allow_html=True)
        st.metric("mean depth", f"{sum(vals)/max(len(vals),1):.2f} / {cfg.n_layer}")
