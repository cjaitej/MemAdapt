# `agpt/` — AdaptiveGPT

A GPT that learns, per token, how many transformer layers to spend on it. Easy tokens
exit early and are unembedded from wherever they stopped; hard tokens run the full
stack. An exited token is still part of the context — later tokens keep attending to
its frozen state.

Research framing: [`../PROJECT.md`](../PROJECT.md).
Measured findings and deviations: [`../docs/DESIGN_NOTES.md`](../docs/DESIGN_NOTES.md).

## Layout

```
model/
  config.py        AdaptiveGPTConfig + the four arms (dense/random/fixed/adaptive)
  router.py        ExitRouter: Linear(d,64)->ReLU->Linear(64,1)->sigmoid, and the
                   straight-through gate that is the routers' only gradient path
  blocks.py        transformer blocks; `delta` for the gated path, `delta_compact`
                   for the subset-of-positions path that makes the saving real
  adaptive_gpt.py  the model. TWO forward paths -- read the module docstring
  targets.py       delta / KL convergence labels derived from a dense pass
  flops.py         analytic cost model, validated exactly against torch's profiler
  retrofit.py      pretrained GPT-2 -> this stack, and the freeze
data/
  prepare.py       WikiText-103 / FineWeb-Edu / local -> flat *_tokens.npy shards
  loaders.py       SegmentLoader: contiguous T-token segments, resumable
  synthetic.py     random shards, for plumbing checks with no corpus
train.py           the three-stage trainer
evaluate.py        the measurement library every script reports from
precision.py       bf16 on Ampere+, fp16 + GradScaler elsewhere
```

## The two forward paths

This is the thing to understand before changing anything here.

`AdaptiveGPT.forward` runs every block over every token and **gates** each block's
contribution. Static shapes, full gradient, compile-friendly — and no faster than
dense, because the arithmetic still happens and is then multiplied by zero. This is the
training path.

`AdaptiveGPT.forward_compact` gathers the surviving tokens at each layer and runs the
block on that subset alone. Dynamic shapes, inference only, genuinely faster. This is
what the benchmarks time and what `generate` uses.

**They compute the same function** — agreement measured at 2.4e-07 max absolute logit
difference, i.e. fp32 rounding. `tests/test_exit.py::test_compact_matches_dense` holds
them to it across bucket sizes and both `exited_as_keys` policies, and it is the only
thing standing between "the fast path is faster" and "the fast path is faster because
it is computing something else".

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r ../requirements.txt
pip install triton-windows          # Windows only; REQUIRED for torch.compile
```

## Run

```bash
pytest -q                                              # 74 tests

python -m agpt.data.prepare --dataset wikitext103
python -m agpt.train --stage dense  --run-name s1_dense --compile
python -m agpt.train --stage routers --init-from runs/s1_dense/best.pt \
    --run-name s2_routers --compile
python -m agpt.train --stage joint  --init-from runs/s2_routers/best.pt \
    --run-name s3_joint --lambda-depth 0.05 --compile

python -m agpt.train --stage dense --synthetic --max-steps 20   # plumbing check
```
