# Training on Kaggle's free GPUs

Yes, this runs on Kaggle free tier. Three things had to change first, and one
limitation remains.

## What Kaggle gives you

| | |
|---|---|
| GPU | Tesla T4 (16 GB), T4 x2, or P100 (16 GB) |
| VRAM | **16 GB vs the 4 GB this was developed on** |
| Session cap | 12 h interactive; "Save & Run All" runs headless |
| Weekly quota | ~30 GPU-hours (Kaggle changes this — check your account) |

**Pick the single T4.** Its fp16 tensor cores beat the P100, which has none. The
`T4 x2` option is wasted here: `amt/train.py` has no DDP wiring, so the second GPU
sits idle.

## The blocker: no bf16

bf16 needs compute capability **≥ 8.0** (Ampere). The T4 is Turing (7.5) and the P100
is Pascal (6.0). Neither supports it, so the original hardcoded `torch.bfloat16` would
have failed or dropped to an emulation path slow enough to look like a hang.

`amt/precision.py` now auto-detects: bf16 on Ampere+, **fp16 + `GradScaler`**
elsewhere. The scaler is not optional — fp16's exponent range is much narrower than
fp32's, so without loss scaling small gradients flush to zero and the model quietly
trains worse rather than failing. `select_precision` returns the dtype and the scaler
flag together so they cannot be mismatched.

You do not need to pass anything: `--precision auto` is the default. Force it with
`--precision fp16` to reproduce Kaggle behaviour locally.

## Setup

**1. Upload the data as a Kaggle Dataset.** Your `data/fineweb_edu_docs/` is 0.62 GB —
upload it rather than re-tokenising on Kaggle (that costs GPU-quota hours doing
CPU work). It mounts read-only at `/kaggle/input/<dataset-name>/`.

**2. Notebook settings:** Accelerator → GPU T4; Internet → On (needed for pip, and it
requires phone verification on Kaggle).

**3. First cell:**

```python
!git clone -b amt https://github.com/<you>/build-nanogpt.git /kaggle/working/amt-repo
%cd /kaggle/working/amt-repo
!pip install -q tiktoken            # torch, numpy, tqdm are preinstalled
!python -m pytest tests/ -q         # 45 tests; confirms the environment is sane
```

Triton ships with Linux PyTorch, so `--compile` works out of the box — no
`triton-windows` needed.

**4. Train:**

```python
!python -m amt.train \
    --variant b6_amt_joint --compile \
    --data-dir /kaggle/input/fineweb-edu-docs \
    --out-dir /kaggle/working/runs \
    --batch-size 24 --max-steps 4500 --run-name r1_b6
```

**Raise the batch size.** You have 16 GB instead of 4. `--batch-size 24` (or 32) cuts
gradient-accumulation steps and improves utilisation substantially. Peak VRAM was
2.5 GB at B=8 locally, so B=24 lands near 7 GB with room to spare.

## Surviving the 12-hour cap

Checkpoints carry model, optimizer, GradScaler **and dataloader stream state**, so a
resume continues on the exact next segment rather than restarting the document:

```python
!python -m amt.train --variant b6_amt_joint --compile \
    --data-dir /kaggle/input/fineweb-edu-docs \
    --out-dir /kaggle/working/runs --run-name r1_b6 \
    --max-steps 4500 --resume latest
```

Write checkpoints to `/kaggle/working/` (persists when you commit the notebook) and
set `--ckpt-every 500` so a killed session loses at most a few minutes.

## Expected time

Locally the RTX 3050 does 25.5k tok/s compiled. A T4 on fp16 should land somewhere
around 1.3–1.8x that — call it 35–45k tok/s, but **measure it before planning**:

```python
!python scripts/benchmark.py --compile --batch-size 24
```

At ~40k tok/s one 300M-token epoch is roughly 2 hours, so the six-variant matrix is
~12–15 h — inside one week's quota. Treat these as estimates until the benchmark
prints real numbers.

## Caveats

- **No DDP.** Single GPU only; `T4 x2` gains nothing.
- **fp16 is not bf16.** If you see loss spikes or NaNs that never appeared locally,
  suspect the scaler before the model. `--precision fp32` is the slow diagnostic.
- **Compile cost.** ~4 min locally per graph, likely less on Kaggle's faster CPU. Each
  capacity-anneal level triggers one recompile; keep `--capacity-levels` small for
  short runs.
- **Quota is wall-clock, not compute.** A session left idle still burns it. Use
  "Save & Run All" for long runs rather than an open browser tab.
