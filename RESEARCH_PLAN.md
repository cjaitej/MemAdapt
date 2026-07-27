# Adaptive Memory Transformer (AMT)

**End-to-end project plan** — built on this nanoGPT fork.

> **Status: weeks 1–5 of §10 are implemented and passing.** The `amt/` package, the
> test suite (45 tests), the document-aware loader, the training loop and the
> benchmark harness all exist and run on the target GPU.
>
> Four findings from the build changed decisions in this plan. They are recorded in
> [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) and flagged inline below:
> 1. `torch.compile` (via `triton-windows`) is **mandatory** — eager-mode routing is
>    *slower* than dense. Compiled, it is 1.06–1.08x faster.
> 2. The output head is **44% of FLOPs** at this width, which shortens the Pareto axis
>    and argues for reporting non-embedding FLOPs.
> 3. The §2.4 budget loss has **zero gradient** under fixed-capacity routing and was
>    replaced by an iso-FLOP sweep.
> 4. Measured throughput is **~10.9 h per 1B tokens**, confirming the §10 timeline.

---

## 0. Verdict: is this possible with the existing code?

**Yes.** `train_gpt2.py` is a clean, single-file GPT-2 implementation with the exact seams
this project needs:

| Needed change | Where it hooks in |
|---|---|
| Adaptive depth | the block loop in `GPT.forward` ([train_gpt2.py:120-121](train_gpt2.py#L120-L121)) |
| Memory read | inside `CausalSelfAttention.forward` ([train_gpt2.py:26-40](train_gpt2.py#L26-L40)) |
| Routers | new modules, consuming `x` in `Block.forward` ([train_gpt2.py:66-69](train_gpt2.py#L66-L69)) |
| Aux losses | `GPT.forward` returns `(logits, loss)` — extend to return a loss dict ([train_gpt2.py:125-128](train_gpt2.py#L125-L128)) |
| Doc-aware batching | `DataLoaderLite` ([train_gpt2.py:214-252](train_gpt2.py#L214-L252)) — **must be rewritten**, see §5 |

Two things must change that are *not* obvious from the diagram:

1. **The dataloader is the hardest single piece of work.** `DataLoaderLite` chops a flat
   concatenated token stream with no notion of document boundaries. External memory is
   meaningless without documents: you need long docs split into *consecutive* segments,
   with the memory bank carried across segments within a document and **cleared at
   document boundaries**. Get this wrong and your memory metrics are silently garbage.
2. **Naive dynamic routing is slower than dense.** If you implement "run a variable number
   of layers per token" with Python control flow and ragged tensors, you get a model that
   uses fewer FLOPs and runs *slower*. The fix (fixed-capacity top-k routing, §2.2) is
   non-negotiable and shapes the whole architecture.

---

## 1. Research framing

### 1.1 Question

> Can a small language model learn to jointly allocate **computation** (depth) and
> **external memory** (retrieval) per token, and does allocating them *jointly* beat
> allocating either one alone at matched FLOPs?

### 1.2 Central hypothesis (this is the actual contribution)

**Retrieval and computation are substitutes.** A token whose continuation is recoverable
from memory (a recurring proper noun, a variable name defined 3000 tokens ago, a repeated
API signature) needs *lookup*, not *reasoning* — it should retrieve and exit early. A token
that requires composition needs depth and gains nothing from retrieval.

If true, a router that sees retrieval outcomes before deciding depth should dominate any
router that decides them independently. That is a falsifiable claim with a clean figure:
**allocated depth vs. retrieval confidence**, plus a **quality-vs-FLOPs Pareto curve**.

### 1.3 Honest positioning vs. prior work

You must state this explicitly in the report — an examiner or interviewer will find it in
five minutes, and pre-empting it is what separates a research project from a hobby project.

| Component | Prior work | Your delta |
|---|---|---|
| Adaptive depth | Mixture-of-Depths (Raposo 2024), CALM, PonderNet, Universal Transformer/ACT, Depth-Adaptive Transformer, LayerSkip | Not novel — **you reimplement it as a baseline** |
| kNN memory | Memorizing Transformers (Wu 2022), kNN-LM, RETRO, Landmark Attention | Not novel — **you reimplement it as a baseline** |
| **Coupling the two under one budget** | — | **This is the contribution** |

Your novel claims, narrowly stated:

- **C1 (mechanism):** *retrieval-conditioned depth allocation* — the depth router is
  conditioned on the outcome of the memory read, so retrieval success can buy back compute.
- **C2 (objective):** a **single joint FLOPs budget** spanning depth and retrieval, which
  forces the model to discover the compute↔memory exchange rate rather than having it
  hand-set.
- **C3 (empirical):** measurement of that exchange rate, and evidence that the joint
  allocator Pareto-dominates depth-only and memory-only allocation at matched FLOPs.

C3 is the one that survives even if C1/C2 turn out to be minor. **A negative result here is
still publishable and still a good project** — "we measured the compute–memory exchange rate
and found retrieval does *not* substitute for depth below X params" is a real finding. Plan
your writeup so it works either way.

---

## 2. Architecture specification

### 2.1 Overall layout (12 layers, d=384, 6 heads)

```
tokens
  │
  ├─ embed (wte + wpe)
  │
  ├─ Layers 0-2      DENSE TRUNK (always run — builds routing features)
  │
  ├─ Layer 3         MEMORY LAYER
  │                    ├─ Memory router  → which tokens query the bank
  │                    ├─ kNN read       → top-k from bank
  │                    ├─ Gated fusion   → y = g·y_mem + (1-g)·y_local
  │                    └─ emits conf_t   (retrieval confidence)
  │                                          │
  ├─ Layers 4-10     ADAPTIVE STACK ◄────────┘
  │                    per layer: depth router r_t = σ(w·[x_t ; conf_t])
  │                    top-⌈c·T⌉ tokens run the block; rest pass through
  │
  ├─ Layer 11        DENSE (always run — stabilises the output distribution)
  │
  └─ ln_f → lm_head
```

Trunk and final layers stay dense on purpose: routing on raw embeddings is uninformed, and
skipping the last layer wrecks calibration. This is a standard finding — cite it, don't
rediscover it.

### 2.2 Depth router — fixed-capacity top-k (the critical implementation detail)

For each adaptive layer ℓ with capacity fraction `c_ℓ ∈ (0,1]` and sequence length `T`:

```python
r = torch.sigmoid(self.w_route(feat)).squeeze(-1)     # (B, T) router scores
k = math.ceil(c * T)                                   # STATIC k -> static shapes
idx = r.topk(k, dim=1).indices.sort(dim=1).values      # (B, k), sorted = causality preserved
xs  = torch.gather(x, 1, idx[..., None].expand(-1, -1, C))
ys  = block(xs)                                        # dense block, k tokens
x   = x.scatter_add(1, idx[...,None].expand(-1,-1,C),
                    r.gather(1, idx)[..., None] * ys)  # residual, router on grad path
```

Three things this buys you, each of which is a trap if missed:

1. **Static shapes.** `k` is a compile-time constant. `torch.compile` works, no
   recompilation, no ragged kernels. This is why the model is actually faster, not just
   theoretically cheaper.
2. **Router gets gradient.** Multiplying the block output by `r` puts the router scalar on
   the backward path. Without this the router receives no learning signal at all and stays
   at its initialisation forever. This is the single most common way this class of model
   silently fails.
3. **Sorted indices** keep original token order, so `F.scaled_dot_product_attention(...,
   is_causal=True)` remains correct inside the block. Attention within an adaptive block is
   among *selected tokens only* — that is intended (it's what makes it cheap), but say so
   in the report, because it means routed-out tokens are invisible as keys at that layer.

**Causality violation at training time.** `topk` over the sequence dimension peeks at the
future — token 5's selection depends on token 900's score. This is fine for teacher-forced
training but **impossible at autoregressive inference**. The standard fix, which you must
implement:

- Train a small auxiliary head `p_t = σ(u · sg[x_t])` with BCE against the label
  `1[t ∈ top-k]` (stop-gradient into the trunk so it can't distort the LM).
- At inference, route token `t` iff `p_t > τ_ℓ`, with `τ_ℓ` calibrated per layer on a
  held-out set so the realised capacity matches `c_ℓ`.
- **Report the train/inference routing agreement rate.** If it's below ~85% your inference
  numbers are not measuring the model you trained, and you need to say so.

### 2.3 Memory: kNN over the model's own KV cache

Not a text/RAG index — a non-parametric bank of the model's own attention keys/values from
earlier segments of the same document (Memorizing-Transformer style). This keeps the whole
thing trainable, dependency-free, and directly measurable by perplexity.

**Bank.** Per batch element, FIFO of `M` entries of `(k, v)` taken from layer 3, `detach()`ed
(no gradient into memory — you are not backpropagating through history). Cleared on document
boundary.

**Read.** For routed query tokens only:

```
sim   = normalize(q) @ normalize(K_mem).T        # (B, nh, k_routed, M)
top   = sim.topk(n_nbr, dim=-1)                  # n_nbr = 32
y_mem = softmax(top.values / sqrt(hs)) @ gather(V_mem, top.indices)
conf  = top.values[..., 0].mean(dim=1)           # (B, k_routed) — feeds the depth router
```

**Fusion.** Per-head learned gate, conditioned on the token and on retrieval quality:

```
g   = sigmoid(b_h + W_g @ [x_t ; conf_t])        # b_h init to -2.0, see §9 risk R2
y   = g * y_mem + (1 - g) * y_local
```

**Memory-router → depth-router coupling (contribution C1).** `conf_t` is broadcast to all
adaptive layers and concatenated into the depth-router input. Tokens that did not retrieve
get `conf_t = 0` and a learned "no-retrieval" embedding. This is the wire that lets the
model learn "good hit ⇒ fewer layers."

**VRAM warning.** The `sim` tensor is `B × nh × T × M`. At B=4, nh=6, T=512, M=8192 that is
100M entries ≈ 200 MB in bf16 — on a 4 GB card that alone can OOM you. Two mitigations, use
both: chunk queries into blocks of 128, and exploit the fact that only routed tokens query
(25% routing ⇒ 4× cheaper). Start training with M=2048 and scale M at eval time.

### 2.4 Loss

> **Superseded during implementation.** The budget term below has *zero gradient* under
> fixed-capacity routing: with `k = ceil(c·T)`, realised FLOPs are a deterministic
> function of the config, so there is nothing for the loss to optimise. The budget is
> now enforced architecturally and the exchange rate is measured by an iso-FLOP sweep
> (`FlopModel.iso_flop_configs`). See [DESIGN_NOTES §3](docs/DESIGN_NOTES.md).
> Contribution C1 is unaffected; C2 is restated as a measured constraint rather than a
> learned one. Bonus: this largely eliminates risk R1.

As implemented:

```
L = L_LM
  + λ_a · L_aux_BCE                    # causal routing predictor
  − λ_e · H(r)                         # entropy bonus, keeps router gradients alive
```

Starting values: `λ_a = 0.05`, `λ_e = 0.01 → 0` over the first 40% of steps.

Originally planned (kept for the writeup's methods section):

```
  + λ_b · (F̂/F_target − 1)²          # joint compute+memory FLOPs budget  (C2)
```

---

## 3. Scale plan — what actually fits in 4 GB

Do not attempt GPT-2 124M training. Karpathy's run was 8×A100 for ~2 hours; on a 3050 it is
weeks and will OOM on the logits tensor regardless.

**AMT-small (the research model):**

| | |
|---|---|
| n_layer | 12 (needs depth for adaptive depth to mean anything) |
| n_embd | 384 |
| n_head | 6 (head size 64) |
| block_size | 512 (segment length; docs span many segments) |
| vocab | 50304 (keep GPT-2 BPE — reuses `fineweb.py`, `hellaswag.py`, and enables §8 retrofit) |
| params | ~21M non-embedding + 19M tied embedding ≈ **40M** |
| micro-batch B | 4 |
| grad accum | 32 → ~65k tokens/optimiser step |
| precision | bf16 autocast (Ampere — supported) |

**Memory budget check:** params+grads+Adam ≈ 640 MB · activations ≈ 200 MB · logits
(4×512×50304 bf16) ≈ 206 MB + the fp32 copy inside `F.cross_entropy`. That last one is the
sneaky term — **use a chunked cross-entropy** (compute loss in 4 slices over the batch) or
you will OOM at exactly the wrong moment.

**Throughput estimate:** ~6·N·D FLOPs ⇒ 1B tokens ≈ 2.4e17 FLOPs; at a realistic 6–10
effective TFLOPS that is **~7–11 hours per 1B-token run**. Budget accordingly:

- Ablations / sweeps: **300M tokens** (~3 h each)
- Final headline runs: **1.5B tokens** (~12 h each, run overnight)

Six final runs ≈ 3 nights. That is the real project budget. Plan §10 around it.

---

## 4. Repo structure

Refactor out of the single file, but stay nanoGPT-flavoured (readable, no framework):

```
amt/
  model/
    blocks.py        # Block, CausalSelfAttention (from train_gpt2.py, +memory hook)
    routers.py       # DepthRouter, MemoryRouter, causal aux predictor
    memory.py        # KVMemoryBank: write/read/clear/FIFO, chunked kNN
    amt.py           # AMTConfig, AMT (assembles trunk/memory/adaptive/dense)
    flops.py         # analytic FLOP model, differentiable + exact accounting
  data/
    prepare.py       # from fineweb.py, but preserves document boundaries
    loaders.py       # DocSegmentLoader (§5) + flat loader for baselines
  train.py           # from train_gpt2.py loop, + aux losses, + W&B
  evals/
    ppl.py  position_ppl.py  recall_probe.py  hellaswag.py  efficiency.py
  app/               # §8 demo
configs/
  dense.yaml  mod_only.yaml  mem_only.yaml  amt_joint.yaml  random_route.yaml
scripts/
  run_sweep.ps1  make_figures.py
docs/
  REPORT.md
```

Keep the original `train_gpt2.py` untouched on `master` as the reference baseline. Do all
work on a `amt` branch.

---

## 5. Data plan (do this early — it gates everything)

**Corpus:** FineWeb-Edu 10B, but only the first ~15 shards (~1.5B tokens, ~3 GB). Full 10B
is 20 GB and you will never train on it.

**The required change to `fineweb.py`:** it currently writes flat concatenated token shards
with `<|endoftext|>` delimiters and no index. You need, per shard:

- `tokens.npy` — the token stream (as now)
- `doc_offsets.npy` — start index of every document
- filter to documents **≥ 2048 tokens** (4+ segments) — short docs make the memory
  mechanism untestable, and most of FineWeb is short

**`DocSegmentLoader`** replaces `DataLoaderLite`:

- Maintains `B` independent "streams", each parked on one long document
- Each `next_batch()` returns the *next consecutive* 512-token segment of each stream
- Returns a `reset_mask` of shape `(B,)` flagging streams that just started a new document
  → training loop calls `memory.clear(reset_mask)`
- Must be DDP-shardable and deterministic-resumable (save stream state in the checkpoint)

**Second corpus for long-range eval:** PG-19 or arXiv (documents are 10k–100k tokens).
FineWeb docs are too short to show off memory. This is where your memory results will
actually appear.

**Synthetic probe corpus:** generated key–value recall task — `"<key_37> is <val_812>"`
early in a long context, queried thousands of tokens later. Gives you a *direct* retrieval
accuracy number instead of inferring memory usefulness from perplexity deltas. Cheap to
build, disproportionately convincing in a writeup.

---

## 6. Training recipe

Inherit nanoGPT's schedule (AdamW, β=(0.9,0.95), wd=0.1, grad-clip 1.0, cosine with warmup)
with these changes:

1. **Phase 1 — dense warmup (first 10% of steps).** All capacities `c_ℓ = 1.0`, memory gate
   forced open, `λ_b = 0`. Builds usable representations before routing decisions matter.
2. **Phase 2 — budget anneal (10%→40%).** Linearly anneal `F_target` from dense down to the
   target (e.g. 0.5× dense FLOPs). Ramp `λ_b` in. **Anneal, never jump** — a hard budget
   from step 0 collapses the router.
3. **Phase 3 — steady state (40%→100%).** Fixed budget, `λ_e → 0`, aux predictor training on.
4. Router params in a **separate optimiser group with 10× lower LR** and **no weight decay**
   (they're small and touchy — decaying them pulls them to the collapse point).

**Instrument from day one** (W&B or a CSV + matplotlib): mean layers/token, per-layer routing
rate, retrieval rate, mean gate value, mean `conf`, routing entropy, aux-predictor agreement.
When a run fails you will diagnose it from these curves, not from the loss.

---

## 7. Evaluation protocol

### 7.1 Baselines (all matched on **FLOPs**, not params — this is the comparison that counts)

| # | Model | Purpose |
|---|---|---|
| B1 | Dense AMT-small, no routing, no memory | the control |
| B2 | Dense + always-on memory (≈ Memorizing Transformer) | is memory worth anything here? |
| B3 | Depth routing only (≈ MoD) | is adaptive compute worth anything here? |
| B4 | **Random routing at matched budget** | **proves the router learned something** |
| B5 | Smaller dense model at matched FLOPs | "just train a smaller model" rebuttal |
| B6 | **AMT joint (yours)** | the claim |
| B7 | AMT with the coupling wire cut (`conf` not fed to depth router) | **isolates contribution C1** |

B4 and B7 are the two ablations that make or break credibility. Do not skip them.

### 7.2 Quality metrics

- Val perplexity (FineWeb-Edu held out)
- **Perplexity bucketed by position-in-document** (segment 1 vs 2 vs 5 vs 10+) — the curve
  where memory should visibly help and dense should be flat
- Perplexity on **repeated-entity tokens** (tokens whose string appeared earlier in the doc)
  vs. first-occurrence tokens — the sharpest slice for the memory claim
- Long-range PPL on PG-19/arXiv
- Synthetic recall accuracy @ distance {1k, 4k, 16k, 64k}
- HellaSwag (already implemented in [hellaswag.py](hellaswag.py); expect near-chance at 40M —
  report it, don't oversell it)

### 7.3 Efficiency metrics

FLOPs/token (analytic **and** `torch.utils.flop_counter.FlopCounterMode` for ground truth),
mean layers/token, retrieval rate, tokens/sec, latency p50/p95 at batch 1, peak VRAM.

### 7.4 FLOP accounting (state the formula in the report)

Per token, per layer, forward:  `F_layer ≈ 24 d² + 4 d T_ctx`
Retrieval (exact kNN + attend):  `F_mem ≈ 2 d M + 2 d n_nbr`

Exchange rate **ρ = F_mem / F_layer** — at d=384, T=512, M=2048: ρ ≈ 0.4, i.e. *one retrieval
costs about 0.4 layers*. If the learned router recovers a trade-off near this ratio, that is
a headline result. Verify the analytic model against the profiler and report the discrepancy.

### 7.5 The money figures

1. **Pareto:** val PPL vs FLOPs/token, one curve per baseline, budget swept over
   {0.3, 0.5, 0.7, 1.0}× dense. The claim is that B6 sits below B2/B3/B5.
2. **Coupling evidence:** mean allocated depth vs. retrieval confidence, binned. A downward
   slope is direct evidence for C1. (A flat line falsifies it — report that honestly.)
3. **Where the compute goes:** routing rate by token type (function words, rare nouns,
   code identifiers, first-vs-repeat occurrence). This is the interpretability figure and
   the one people remember.

---

## 8. Demo application — the honest version

**A 40M-param model trained on 1.5B tokens will not be a useful coding assistant.** Shipping
"personal AI coding companion" on top of it would be the weakest part of the project and the
first thing a sharp interviewer would poke. Two credible options:

**Option A (recommended) — Retrofit onto GPT-2 124M.** Load pretrained GPT-2 via the existing
`GPT.from_pretrained` ([train_gpt2.py:130-177](train_gpt2.py#L130-L177)), freeze the backbone,
insert the memory layer + routers, and train **only the new modules** (a few M params) on
your own code corpus. Fits in 4 GB. The demo is then:

> A GPT-2 that has read your repository, retrieves from it token-by-token, and skips layers
> on the tokens it can just look up.

Measurable: perplexity on held-out files from your own repo, base vs retrofit; plus latency
and mean-layers/token. This is a *real* demo with *real* numbers and it directly re-uses the
research contribution.

**Option B — scope the demo to what the model can do.** Next-token completion in a small
editor UI with a live "compute/memory allocation" visualiser: a heatmap over the generated
text showing layers used and retrieval hits per token. Weak as a product, excellent as a
demonstration of the mechanism.

Do A as the deliverable and B as the visualisation inside it.

---

## 9. Risk register

| | Risk | Signal | Mitigation |
|---|---|---|---|
| R1 | ~~**Router collapse**~~ **largely retired** | — | fixed capacity means the rate cannot drift; entropy bonus retained only to keep router gradients alive |
| R2 | **Memory ignored** (gate → 0) | mean gate < 0.05 | gate bias init −2.0 *plus* an annealed `gate_floor` holding it open during warmup (implemented); if still dead, your docs are too short (§5) |
| R3 | **Slower than dense** despite fewer FLOPs — **CONFIRMED, then fixed** | tokens/sec drops | **measured at 0.82–0.98x in eager mode.** `torch.compile` (needs `pip install triton-windows`) turns it into 1.06–1.08x. Never benchmark eager |
| R4 | **Train/inference routing mismatch** | aux agreement < 85% | more aux capacity, per-layer threshold calibration; report the gap |
| R5 | **OOM on 4 GB** | — | chunked CE, chunked kNN, grad checkpointing on adaptive blocks, B=2 fallback |
| R6 | **Memory leaks future info** | suspiciously low PPL | assert bank only contains segments strictly before current; write a unit test for this — it is the classic way retrieval-LM papers get retracted |
| R7 | **No effect at 40M scale** | B6 ≈ B3 ≈ B1 | this is a *result*: report the exchange rate and the scale at which it fails to pay off. Frame the project around measurement, not around a win |

R6 deserves a dedicated test in CI. Write it in week 4, not week 12.

---

## 10. Timeline (14 weeks)

Each phase ends with something that runs.

| Wk | Phase | Deliverable |
|---|---|---|
| 1 | ~~Setup~~ **done** | PyTorch 2.11+cu128 + `triton-windows`, GPU verified, throughput recorded |
| 2 | ~~Data~~ **code done** | `prepare.py` with doc offsets, `DocSegmentLoader` + 11 tests. *Corpus not yet downloaded* |
| 3 | ~~Refactor~~ **done** | `amt/` package, `AMTConfig` variants B1–B7, JSONL logging, FLOP model validated vs profiler |
| 4 | ~~Memory~~ **done** | `KVMemoryBank`, chunked kNN read, gated fusion, **R6 leakage test green** |
| 5 | ~~Depth~~ **done** | `TopKTokenRouter`, aux causal predictor, causal path prefix-invariant |
| 6 | Integration | **All 7 variants construct, train and backward.** Remaining: real-corpus runs |
| 7 | Stabilise | Fix collapse (R1/R2), tune λ's, get B6 training reliably at 3 budget settings |
| 8 | Evals | All of §7.2/§7.3 implemented, synthetic recall probe, FLOP counter validation |
| 9 | Ablations | **B4** and **B7** (the credibility runs) + B5 |
| 10 | Final runs | 1.5B-token runs for B1/B2/B3/B6 at the chosen budget (~3 nights) |
| 11 | Analysis | The three figures in §7.5, exchange-rate measurement |
| 12 | Demo | Option A retrofit onto GPT-2 124M over your own repo, with §8 numbers |
| 13 | Demo UI | Allocation visualiser, packaging, README with reproduce-me instructions |
| 14 | Write-up | `docs/REPORT.md` in paper form, figures, limitations section |

**Weeks 6–7 are the risky ones.** If integration is still broken at end of week 7, cut scope
to: memory-only + depth-only + a *post-hoc* coupling analysis, and reframe C1 as future work.
Decide that at the week-7 checkpoint, not in week 12.

---

## 11. Next concrete steps

Weeks 1–5 are built (see the status box at the top). What is left before the first
real result:

1. **Download the corpus** — the only thing blocking a real training run:
   ```
   pip install datasets
   python -m amt.data.prepare --shards 3 --min-doc-tokens 2048
   ```
   Expect ~3 GB and a few hours. Check the reported drop rate: most of FineWeb is
   shorter than 2048 tokens, and if almost everything is dropped, lower the threshold.
2. **Train B1 for ~2000 steps** and confirm the loss curve looks like nanoGPT's:
   ```
   python -m amt.train --variant b1_dense --compile --max-steps 2000
   ```
3. **Train B6** and watch the instrumentation, not the loss: routing rate, mean gate,
   `mem_conf_mean`, aux agreement. Those diagnose failures; the loss will not.
4. **Long-document corpus** (PG-19 or arXiv) for the eval where memory should actually
   show up — FineWeb documents are too short.
5. **Build the eval suite** (§7.2): position-bucketed perplexity, repeated-entity
   perplexity, recall probe scoring.

Already handled: chunked cross-entropy (`AMTConfig.ce_chunks`), the OOM you were
warned about — peak VRAM is 2.5 GB at B=8, leaving headroom.

---

## 12. What this gives you on a resume

- **Research contribution:** Adaptive Memory Transformer — joint per-token allocation of
  depth and retrieval under a unified FLOPs budget; measurement of the compute–memory
  exchange rate in small LMs.
- **Engineering:** custom transformer variant, fixed-capacity routing kernels, non-parametric
  KV memory with kNN retrieval, document-aware streaming dataloader, FLOP accounting,
  4 GB-VRAM training discipline.
- **Evaluation:** matched-FLOPs Pareto analysis across 7 model variants, with the two
  ablations (random routing, cut coupling) that make the result trustworthy.
- **Application:** adaptive-memory retrofit of GPT-2 that personalises to a codebase, with
  measured perplexity and latency deltas.

The thing that makes this stronger than "another RAG chatbot" is §7.1 B4 and B7 and the
willingness to report R7. Keep them in.
