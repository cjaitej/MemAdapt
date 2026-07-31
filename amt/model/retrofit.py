"""Load pretrained GPT-2 into the AMT stack, then freeze everything it brought.

    from amt.model.retrofit import gpt2_config, from_gpt2, freeze_backbone

    cfg = gpt2_config("gpt2", block_size=1024)
    model, report = from_gpt2("gpt2", cfg)
    freeze_backbone(model)

Why this is a rename and not a port
-----------------------------------
`blocks.py` kept nanoGPT's parameter names, which are HuggingFace's names for GPT-2,
so every tensor in `GPT2LMHeadModel` has exactly one counterpart here. Two mechanical
differences, and nothing else:

* `AdaptiveBlock` wraps the real block, so its tensors sit one level deeper under
  `.block.`. That is the only key rewriting this does.
* OpenAI's checkpoints store the four projection weights as `Conv1D`, which is a
  Linear with its axes swapped, so those four are transposed on the way in.

What AMT adds -- the routers, the fusion gate, the softmax temperature -- has no GPT-2
counterpart and keeps its own initialisation. At `gpt2` that is **21,559 parameters,
0.017% of the backbone**. Freezing the rest is what makes "base vs ours" an exact
comparison: both arms run byte-identical pretrained weights, and the only difference
is the routing and retrieval this project contributes.

The vocabulary decision
-----------------------
`gpt2_config` uses vocab_size 50257, not the 50304 the from-scratch configs pad to.
Padding is free when the rows are trained from scratch, but here the extra 47 rows
would arrive randomly initialised into an otherwise-pretrained head, and a softmax
does not know they are not real tokens -- they take probability mass from the ones
that are, which shows up as a perplexity gap that has nothing to do with routing.
Pass `vocab_size=50304` explicitly if you want the tensor-core alignment back and are
willing to pay for it in the comparison.
"""

import re

import torch

from .amt import AMT
from .config import AMTConfig

# n_layer / n_head / n_embd for each OpenAI checkpoint. Same table as
# train_gpt2.py::GPT.from_pretrained, which is where this approach comes from.
GPT2_SHAPES = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),     # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),    # 774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),    # 1558M
}

# Conv1D-stored weights: (in, out) on disk, (out, in) in a Linear.
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight",
              "mlp.c_fc.weight", "mlp.c_proj.weight")

# Buffers, not parameters: the causal mask and its -inf twin.
HF_SKIP = (".attn.bias", ".attn.masked_bias")

# Everything AMT adds on top of GPT-2. Matched against parameter names.
NEW_MODULES = (".router.", ".mem_router.", ".w_gate.", ".log_temp")

# wte and lm_head are the same tensor here (weight tying), and named_parameters()
# yields a shared tensor once -- under whichever name was registered first, which is
# transformer.wte.weight. The HF checkpoint carries both names, so the second must be
# mapped onto the first or it gets reported as an unmatched source tensor: alarming,
# and wrong, since tying means it was already written.
TIED_ALIASES = {"lm_head.weight": "transformer.wte.weight"}


def gpt2_config(model_type="gpt2", **overrides) -> AMTConfig:
    """An AMTConfig shaped like a GPT-2 checkpoint.

    Routing and memory settings keep their AMTConfig defaults, so the retrofit is
    configured exactly like any other variant -- pass `route_depth=False`,
    `use_memory=False`, `couple=False` and so on to build the ablation arms.
    """
    if model_type not in GPT2_SHAPES:
        raise KeyError(f"unknown {model_type!r}; known: {sorted(GPT2_SHAPES)}")
    kw = dict(GPT2_SHAPES[model_type])
    kw.update(vocab_size=50257, block_size=1024, bias=True)
    kw.update(overrides)
    return AMTConfig(**kw)


def _to_amt_key(key, adaptive):
    """HF key -> AMT key. Only adaptive layers move, and only by one level."""
    m = re.match(r"transformer\.h\.(\d+)\.(.+)$", key)
    if m and int(m.group(1)) in adaptive:
        return f"transformer.h.{m.group(1)}.block.{m.group(2)}"
    return key


@torch.no_grad()
def load_gpt2_state_dict(model, hf_state):
    """Copy a GPT2LMHeadModel state dict into `model`, in place.

    Takes the state dict rather than downloading one so the mapping is testable
    without a network round trip -- see tests/test_retrofit.py.

    Source dimensions are read off the tensors, never assumed: a checkpoint with a
    shorter position table or a smaller vocabulary than the target config is a
    legitimate thing to load, and silently mismatching either is the kind of bug that
    surfaces 6 hours into a training run as "the retrofit is worse than the base".
    """
    cfg = model.config
    adaptive = set(cfg.adaptive_layers)
    target = dict(model.named_parameters())
    loaded, missing = [], []

    src_vocab, src_embd = hf_state["transformer.wte.weight"].shape
    src_pos = hf_state["transformer.wpe.weight"].shape[0]
    if src_embd != cfg.n_embd:
        raise ValueError(f"checkpoint width {src_embd} != config n_embd {cfg.n_embd}")
    if cfg.vocab_size < src_vocab:
        raise ValueError(
            f"config vocab_size {cfg.vocab_size} is smaller than the checkpoint's "
            f"{src_vocab}; the pretrained rows would not fit")
    if cfg.block_size > src_pos:
        raise ValueError(
            f"config block_size {cfg.block_size} exceeds the checkpoint's {src_pos} "
            "learned positions; GPT-2 has no embedding to copy for the rest")

    for key, value in hf_state.items():
        if key.endswith(HF_SKIP):
            continue
        name = TIED_ALIASES.get(key, _to_amt_key(key, adaptive))
        if name not in target:
            missing.append(key)
            continue
        param = target[name]

        if any(key.endswith(t) for t in TRANSPOSED):
            value = value.t()
        if name == "transformer.wpe.weight":
            # Keep the first block_size positions. GPT-2's table is ordered, so the
            # head of it is exactly the embedding for a shorter context.
            value = value[:cfg.block_size]
        if name in ("transformer.wte.weight", "lm_head.weight"):
            # wte and lm_head are the same tensor here (weight tying), so this runs
            # twice and writes the same rows twice -- harmless, and cheaper than
            # special-casing which of the two HF happens to emit first.
            if cfg.vocab_size > src_vocab:
                param[:src_vocab].copy_(value)
                loaded.append(name)
                continue

        if param.shape != value.shape:
            raise ValueError(f"{key} -> {name}: {tuple(value.shape)} vs "
                             f"{tuple(param.shape)}")
        param.copy_(value)
        loaded.append(name)

    new = [n for n in target if n not in set(loaded)]
    return {
        "loaded": sorted(set(loaded)),
        "new": sorted(new),
        "unmatched_source": sorted(missing),
        "n_loaded": sum(target[n].numel() for n in set(loaded)),
        "n_new": sum(target[n].numel() for n in new),
        "padded_vocab": cfg.vocab_size - src_vocab,
    }


def from_gpt2(model_type="gpt2", config=None, device="cpu"):
    """Build an AMT at GPT-2's shape and fill it with the pretrained weights."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as e:                              # noqa: BLE001
        raise SystemExit("the 'transformers' package is required for the retrofit:\n"
                         "    pip install transformers") from e

    cfg = config or gpt2_config(model_type)
    expected = GPT2_SHAPES[model_type]
    for k, v in expected.items():
        if getattr(cfg, k) != v:
            raise ValueError(f"config {k}={getattr(cfg, k)} does not match "
                             f"{model_type}'s {v}")

    model = AMT(cfg)
    hf = GPT2LMHeadModel.from_pretrained(model_type)
    report = load_gpt2_state_dict(model, hf.state_dict())
    del hf
    return model.to(device), report


def is_new_module(name):
    """True for a parameter AMT adds on top of GPT-2."""
    return any(marker in name for marker in NEW_MODULES)


def freeze_backbone(model, train_layernorms=False, train_memory_block=False):
    """Train only what the retrofit adds; hold the pretrained weights fixed.

    `train_layernorms` and `train_memory_block` are the first two rungs of the
    mitigation ladder for the risk that a linear router cannot find routing signal in
    features that were never shaped for it. Both are cheap next to the backbone, and
    using either is a result to report rather than a knob to quietly turn: "the
    routers could not learn on frozen features alone" is a finding about the
    mechanism, not about the training setup.
    """
    mem_prefix = f"transformer.h.{model.config.mem_layer}."
    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        train = is_new_module(name)
        if train_layernorms and (".ln_1." in name or ".ln_2." in name
                                 or name.startswith("transformer.ln_f.")):
            train = True
        if train_memory_block and name.startswith(mem_prefix):
            train = True
        param.requires_grad = train
        if train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return {"trainable": trainable, "frozen": frozen,
            "fraction": trainable / max(trainable + frozen, 1)}


def describe(report, freeze=None):
    lines = [
        f"retrofit    : {len(report['loaded'])} tensors from GPT-2 "
        f"({report['n_loaded']/1e6:.1f}M params)",
        f"new modules : {report['n_new']:,} params in {len(report['new'])} tensors",
    ]
    if report["padded_vocab"]:
        lines.append(f"vocab       : {report['padded_vocab']} rows padded past the "
                     "checkpoint and left at init")
    if report["unmatched_source"]:
        lines.append(f"UNMATCHED   : {len(report['unmatched_source'])} source tensors "
                     f"had no target: {report['unmatched_source'][:4]}")
    if freeze is not None:
        lines.append(f"trainable   : {freeze['trainable']:,} of "
                     f"{freeze['trainable'] + freeze['frozen']:,} "
                     f"({freeze['fraction']:.3%})")
    return "\n".join(lines)
