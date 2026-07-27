# Design notes — measured findings and deviations from the plan

Everything here was measured on the target machine (RTX 3050 Laptop, 4 GB, Windows 11,
PyTorch 2.11+cu128) during implementation. Where the build departs from
`RESEARCH_PLAN.md`, the reason is recorded rather than the plan silently rewritten.

---

## 1. `torch.compile` is mandatory, and on Windows it needs `triton-windows`

The single most important measurement of the build. Same model, same batch, only
`torch.compile` toggled:

| variant | eager tok/s | compiled tok/s | compiled speedup vs dense |
|---|---:|---:|---:|
| `b1_dense` | 19,949 | 25,484 | 1.00x |
| `b3_depth_only` | 19,456 | 27,431 | **1.08x** |
| `b6_amt_joint` | 16,298 | 26,923 | **1.06x** |

*(B=8, T=512, 12 steps after 3 warmup.)*

**Eager mode, routing is slower than dense** (0.82–0.98x) despite using 15–16% fewer
FLOPs — risk R3 exactly as predicted. The gather/scatter bookkeeping and the extra
router kernel launches cost more than skipping 3.5 layers per token saves, because at
d=384 on this GPU the model is launch-latency bound, not compute bound.

Compiled, the fixed-capacity static-shape design pays off and routing wins. **Never
benchmark or make an efficiency claim in eager mode.**

Stock PyTorch on Windows ships without Triton, so `torch.compile` raises
`TritonMissing`. Fix:

```
pip install triton-windows
```

This is a hard dependency for any efficiency claim in the project, not a nice-to-have.
If it ever breaks, the fallback is WSL2, where CUDA and upstream Triton both work.

**Conversion is partial and must be reported honestly:** a 15.5% FLOP reduction buys
6% wall-clock. Kernel-launch overhead and the untouched output head absorb the rest.
Quote both numbers; never let the FLOP number stand in for the speedup.

---

## 2. The output head dominates FLOPs, which shortens the Pareto axis

At d=384 with the GPT-2 50257 vocabulary, `lm_head` alone is **44% of dense forward
FLOPs** — about 11x the cost of one transformer layer. Nothing the routers do touches
it, so it sets a floor.

| config | adaptive layers | head | dense total | achievable FLOP range |
|---|---:|---:|---:|---|
| as planned (V=50304, trunk=3) | 7 | 38.6M | 87.4M | 0.67–1.00x |
| V=16384 | 7 | 12.6M | 61.4M | 0.53–1.00x |
| V=16384, trunk=2 | 8 | 12.6M | 61.4M | 0.46–1.00x |
| V=16384, trunk=2, n_layer=16 | 12 | 12.6M | 77.1M | 0.37–1.00x |

**Decision: keep the GPT-2 BPE and report the Pareto on non-embedding FLOPs.**

Reporting non-embedding ("layer") FLOPs is standard in the adaptive-compute
literature, and on that axis the range is a healthy 0.42–1.00x. Keeping the 50k vocab
preserves `hellaswag.py` and — more importantly — the GPT-2 124M retrofit demo, which
needs a matching vocabulary and is the only demo at this scale that is not
embarrassing.

Report **both** axes in the writeup. The head-dominance number is itself a finding
worth stating: at small d, adaptive-depth methods have far less headroom than the
layer-only figures in the literature suggest.

The decision is reversible — `amt/data/prepare.py` takes `--tokenizer`, so switching
is a re-run, not a rewrite.

---

## 3. The budget loss was replaced by an iso-FLOP sweep

`RESEARCH_PLAN.md` §2.4 proposed learning the compute/memory split with a
differentiable budget loss `(F/F_target − 1)²`.

**That loss is meaningless under fixed-capacity routing.** With `k = ceil(c·T)` fixed,
realised FLOPs are a deterministic function of the config — the term has zero gradient
with respect to anything the model can change.

So the budget is enforced architecturally, and the compute↔memory exchange rate is
measured by sweeping `(depth_capacity, mem_capacity)` pairs along an iso-FLOP line
(`FlopModel.iso_flop_configs`). This is strictly better:

- the exchange rate becomes a **measured quantity**, not an optimisation artefact;
- risk R1 (router collapse) is largely eliminated — capacity cannot drift;
- contribution **C1 is untouched**: the router still decides *which* tokens get depth,
  which is the entire content of the coupling claim.

Contribution C2 is therefore restated: not "a learned joint budget" but "a joint
budget constraint under which the allocation is swept and the exchange rate measured".
Weaker as a mechanism, stronger as science.

Analytic exchange rate at the current config: **ρ = 0.406** — one retrieval costs
about 0.4 of a transformer layer.

---

## 4. Training-time routing is non-causal; the causal path is a separate mode

Top-k over the sequence peeks at the future: whether token 5 is selected depends on
token 900's score. Fine under teacher forcing, impossible when generating.

Two modes, and the distinction is load-bearing:

- **`causal=False`** (training, teacher-forced eval): top-k, gather/scatter, fast,
  exactly on budget, not causal.
- **`causal=True`** (generation, causal-mode eval): per-token threshold on the
  auxiliary predictor. Genuinely causal, but the selected count varies, so there is no
  fixed-shape gather — it falls back to a masked dense path that computes the same
  numbers at dense cost.

`AdaptiveBlock._masked_forward` therefore gives **no speedup**. That is deliberate: it
exists to measure the *quality* of causal routing. Speed is measured on the generation
path, where tokens arrive one at a time and the threshold decision is naturally causal
and genuinely cheap.

`tests/test_routing.py` pins both halves: `test_training_mode_is_admittedly_non_causal`
documents the gap, `test_causal_mode_is_prefix_invariant` proves the causal path does
not leak. Always report `route_agreement` alongside any generation result.

---

## 5. Ordering rules that are silent when broken

Two orderings produce no error and no obvious loss-curve symptom when wrong. Both are
pinned by tests.

**Clear the bank BEFORE the forward pass, write AFTER it:**

```python
bank.clear(reset_mask)                    # BEFORE — reset streams start a new doc NOW
_, losses, stats, kv = model(x, y, bank)
model.write_memory(bank, kv)              # AFTER  — invariant R6
```

Clearing after the forward means the batch has already retrieved from the *previous*
document. `AMT.write_memory` deliberately takes no `reset_mask` argument so the
mistake is not available.

**The router weight must multiply the block output.** In `AdaptiveBlock`, `delta` is
scaled by `out["weight"]`. This is the only path by which the router receives
gradient. Remove it and training proceeds normally, the loss goes down, and the
routing decisions stay frozen at their random initialisation forever. Nothing else in
the system reports this — hence `test_router_receives_gradient`.

---

## 6. Measured baseline numbers (record these; every claim is relative to them)

RTX 3050 Laptop 4 GB, B=8, T=512, bf16, compiled:

| | |
|---|---|
| params | 21.3M non-embedding, 40.8M total |
| layout | trunk 0–2, memory layer 3, adaptive 4–10, dense tail 11 |
| dense throughput | 25,484 tok/s |
| joint (AMT) throughput | 26,923 tok/s |
| peak VRAM | 2.5 GB at B=8 (headroom for B=12) |
| **1B tokens** | **~10.9 h dense, ~10.3 h routed** |

This lands inside the plan's 7–11 h estimate, so the §10 timeline stands: ablations at
300M tokens (~3 h), final runs at 1.5B tokens (~15 h, overnight).

**The analytic FLOP model validates** against `torch.utils.flop_counter` to within
2.7–5.5%. The residual is attention: `FlopCounterMode` charges causal attention the
full T×T, while `flops.py` charges the (T+1)/2 a causal mask actually computes. The
analytic model is the more accurate of the two here; the gap is expected and stable.

---

## 7. Three ways to accidentally recompile every step

`torch.compile` takes **~4 minutes** per compile on this GPU, so anything that
invalidates a guard every step makes training effectively impossible — the first
attempt at a 150-step run never reached step 25 in 45 minutes. All three causes were
Python-level values read inside `forward`:

| cause | symptom | fix |
|---|---|---|
| `_stats()` calling `.item()` | graph break + GPU→CPU sync every forward | return **tensors**; convert once outside via `stats_to_floats` |
| `gate_floor` as an annealed float | recompile every warmup step | `register_buffer`, written with `.fill_()` |
| `entropy_weight` as a float argument | recompile every step | buffer + `set_entropy_weight()` |

The general rule: **any scalar that changes during training must be a buffer written
in place, never a Python float or a forward argument.** Tensor identity is stable, so
no guard is invalidated.

Capacity is the deliberate exception — it changes `k`, hence tensor shapes, so a
recompile is unavoidable. That is why `capacity_at` is quantised into levels: each
level costs one recompile (~4 min). Four levels is ~3% overhead on a 10-hour run and
intolerable on a short one, so short debug runs should pass
`--capacity-warmup-frac 0 --capacity-anneal-frac 0`.

## 8. Router thresholds must be calibrated before `agree/*` means anything

`TopKTokenRouter.threshold` starts at 0 and is only set by `calibrate_routers`.
Uncalibrated, the causal predictor classifies every token as routed and
`agreement` returns the *routing rate* — a number that looks like an accuracy,
tracks nothing, and stays plausibly around 0.5.

`evaluate()` now calibrates before measuring. **The bank must be passed to
`calibrate_routers`**: without it the memory read is skipped, so `mem_router` never
runs, never calibrates, and its agreement reports exactly `mem_capacity` (0.25 — the
tell that caught this).

**Currently near chance (0.43–0.51 at 60 steps)** and this is the single most important
number to watch as training scales up. The aux head predicts top-k membership from the
same features the router scores, so it *should* reach high agreement. If it has not
cleared ~0.85 by mid-training, causal generation is running a different model than the
one trained, and every generation-time result needs an asterisk.

## 9. Still open

- **`mem_size` is 2048 during training** to bound the B×H×T×M similarity tensor. Larger
  banks are an eval-time knob; train/eval mismatch at large M is untested.
- **The recall probe measures in-context recall**, which a 40M model will not do
  zero-shot. Mix a fraction of `amt/data/synthetic.py` documents into training, or
  train a dedicated probe run.
- **DDP is untested** — single GPU here. The loader shards documents by rank and the
  bank is per-rank, but none of that has been exercised.
- **FineWeb-Edu has not been downloaded yet.** Everything above is measured on
  synthetic shards, which validates plumbing and throughput but says nothing about
  loss quality.
