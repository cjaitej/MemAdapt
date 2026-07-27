# `amt/` — Adaptive Memory Transformer

A GPT that learns, per token, how much **computation** (depth) and how much **external
memory** (kNN retrieval over its own KV cache) to spend.

Research framing and experiment plan: [`../RESEARCH_PLAN.md`](../RESEARCH_PLAN.md).
Measured findings and deviations: [`../docs/DESIGN_NOTES.md`](../docs/DESIGN_NOTES.md).

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows          # Windows only; REQUIRED for torch.compile
pip install numpy tiktoken tqdm pytest datasets
```

`triton-windows` is not optional. Without `torch.compile`, routing is *slower* than
dense and every efficiency claim inverts. See DESIGN_NOTES §1.

## Run

```bash
pytest tests/ -q                                        # 45 tests

python scripts/benchmark.py --compile                   # throughput + FLOP validation

python -m amt.data.prepare --shards 3                   # tokenise FineWeb-Edu
python -m amt.train --variant b6_amt_joint --compile    # train

python -m amt.train --variant b1_dense --synthetic --max-steps 20   # plumbing check
```

## Layout

```
model/
  config.py    AMTConfig + the B1-B7 experiment matrix as named variants
  flops.py     analytic FLOP model, iso-FLOP sweeps, exchange rate
  memory.py    KVMemoryBank: ring buffer, chunked exact kNN, differentiable read
  routers.py   TopKTokenRouter: fixed-capacity top-k + causal aux predictor
  blocks.py    Block / MemoryBlock / AdaptiveBlock
  amt.py       assembles the stack, chunked cross-entropy, optimiser groups
data/
  prepare.py   tokenise a corpus into document-aware shards (tokens + offsets)
  loaders.py   DocSegmentLoader: consecutive segments + reset_mask
  synthetic.py shard fixtures and the key-value recall probe
train.py       training loop with capacity/gate schedules and instrumentation
```

## Architecture

```
layers 0-2    dense trunk       always run; routing needs contextualised features
layer  3      memory layer      routed kNN read, gated fusion, emits `conf`
layers 4-10   adaptive stack    top-k routed; router conditioned on `conf`  <- C1
layer  11     dense tail        always run; skipping it wrecks calibration
```

The wire from `conf` into the depth routers is the contribution: retrieval success
should let a token skip computation. `b7_uncoupled` cuts that wire and is the ablation
that isolates it.

## Three things that fail silently

1. **Clear the bank before the forward, write after it.** Clearing afterwards lets a
   batch retrieve from the previous document. Pinned by
   `tests/test_memory.py::test_no_future_leakage`.
2. **The router weight must scale the block output.** It is the router's only gradient
   path; without it routing stays at its random init while the loss trains normally.
   Pinned by `test_router_receives_gradient`.
3. **Training-time top-k routing is not causal.** Use `causal=True` for anything
   generation-shaped, and report `agree/*` from the stats dict.

## Variants

| name | routing | memory | purpose |
|---|---|---|---|
| `b1_dense` | — | — | control |
| `b2_memory_only` | — | always | ≈ Memorizing Transformer |
| `b3_depth_only` | top-k | — | ≈ Mixture-of-Depths |
| `b4_random` | random | routed | matched-budget control: did the router learn? |
| `b6_amt_joint` | top-k | routed | the claim |
| `b7_uncoupled` | top-k | routed | coupling wire cut — isolates C1 |
