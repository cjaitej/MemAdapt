"""Non-parametric KV memory bank with exact kNN retrieval.

This is *not* a text/RAG index. The bank stores the model's own attention keys and
values from earlier segments of the same document (Memorizing-Transformers style),
detached from the graph. That keeps the whole mechanism trainable end-to-end through
the read (gradients flow into the query and the gate, never into history) and makes
its benefit directly measurable as perplexity.

THE CRITICAL INVARIANT (risk R6 in RESEARCH_PLAN.md)
----------------------------------------------------
The bank must only ever contain tokens *strictly before* the current segment.
Enforced by ordering: the training loop calls `read()` during the forward pass and
`write()` only *after* the forward pass completes. Never write inside forward.
`tests/test_memory.py::test_no_future_leakage` pins this down; if that test ever
goes red the model's perplexity numbers are meaningless.
"""

import torch
import torch.nn.functional as F


class KVMemoryBank:
    """Per-batch-element ring buffer of (key, value) pairs, with exact kNN search.

    Deliberately not an nn.Module: it holds no parameters and no gradient state, only
    detached buffers. Keeping it out of the module tree stops it being captured in
    state_dict / DDP parameter sync, which would be both wrong and expensive.

    Shapes: keys/values are (B, n_head, capacity, head_dim).
    """

    def __init__(self, batch_size, n_head, head_dim, capacity, device, dtype=torch.bfloat16):
        self.B = batch_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.capacity = capacity
        self.device = device
        self.dtype = dtype

        self.keys = torch.zeros(batch_size, n_head, capacity, head_dim,
                                device=device, dtype=dtype)
        self.values = torch.zeros(batch_size, n_head, capacity, head_dim,
                                  device=device, dtype=dtype)
        # L2-normalised keys, maintained at write time rather than recomputed per
        # read. Normalising the whole bank on every forward pass allocated and wrote
        # a full float32 copy of it each step, which measured as a larger cost than
        # the entire kNN search it was preparing for.
        self.keys_norm = torch.zeros(batch_size, n_head, capacity, head_dim,
                                     device=device, dtype=torch.float32)
        # `fill` counts valid entries (saturates at capacity); `ptr` is the write head.
        self.fill = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.ptr = torch.zeros(batch_size, device=device, dtype=torch.long)

    # -- state management --------------------------------------------------

    def clear(self, mask=None):
        """Drop memory for streams that just moved to a new document.

        `mask` is a (B,) bool tensor; True means "this stream started a new document,
        forget everything". No mask clears every stream.
        """
        if mask is None:
            self.fill.zero_()
            self.ptr.zero_()
            return
        mask = mask.to(self.device, torch.bool)
        self.fill = torch.where(mask, torch.zeros_like(self.fill), self.fill)
        self.ptr = torch.where(mask, torch.zeros_like(self.ptr), self.ptr)

    def detach_(self):
        """Belt-and-braces: the bank must never hold graph references across steps."""
        self.keys = self.keys.detach()
        self.values = self.values.detach()
        self.keys_norm = self.keys_norm.detach()

    @property
    def has_memory(self):
        """(B,) bool -- which streams currently hold at least one entry."""
        return self.fill > 0

    def valid_mask(self):
        """(B, capacity) bool -- which slots hold real data.

        Unwritten slots are zeros, which would otherwise score as similarity 0 and
        could beat genuinely negative similarities.
        """
        ar = torch.arange(self.capacity, device=self.device)
        return ar.unsqueeze(0) < self.fill.unsqueeze(1)

    # -- write -------------------------------------------------------------

    @torch.no_grad()
    def write(self, k, v):
        """Append a segment's keys/values. Call AFTER the forward pass, never during.

        k, v: (B, n_head, T, head_dim). Writes wrap around the ring buffer, so a
        segment longer than the capacity keeps only its most recent `capacity` tokens.
        """
        assert k.shape == v.shape, "key/value shape mismatch"
        B, H, T, D = k.shape
        assert B == self.B and H == self.n_head and D == self.head_dim, (
            f"bank expects (B={self.B}, H={self.n_head}, *, D={self.head_dim}), got {tuple(k.shape)}"
        )
        k = k.detach().to(self.dtype)
        v = v.detach().to(self.dtype)

        if T >= self.capacity:  # only the tail fits
            k, v = k[:, :, -self.capacity:], v[:, :, -self.capacity:]
            T = self.capacity

        # Destination slot for each incoming token, per stream, with wraparound.
        offs = torch.arange(T, device=self.device).unsqueeze(0)          # (1, T)
        pos = (self.ptr.unsqueeze(1) + offs) % self.capacity              # (B, T)
        idx = pos.view(B, 1, T, 1).expand(B, H, T, D)

        self.keys.scatter_(2, idx, k)
        self.values.scatter_(2, idx, v)
        self.keys_norm.scatter_(2, idx, F.normalize(k.float(), dim=-1))

        self.ptr = (self.ptr + T) % self.capacity
        self.fill = torch.clamp(self.fill + T, max=self.capacity)

    # -- read --------------------------------------------------------------

    def _flat_gather(self, source, idx):
        """Gather (B, H, t, k, D) rows out of a (B, H, M, D) bank without expanding it.

        The obvious `torch.gather` formulation needs an index tensor the size of the
        output *and* broadcasts the bank to (B, H, t, M, D) -- billions of elements.
        Flattening to (B*H*M, D) and offsetting the indices keeps the index tensor at
        B*H*t*k int64 and touches only the rows actually retrieved.
        """
        B, H, t, k = idx.shape
        D = source.size(-1)
        flat = source.reshape(B * H * self.capacity, D)
        base = (torch.arange(B * H, device=idx.device) * self.capacity).view(B, H, 1, 1)
        return flat.index_select(0, (idx + base).reshape(-1)).view(B, H, t, k, D)

    def read(self, q, n_neighbors, chunk_size=128, temperature=1.0):
        """Retrieve from memory.

        q: (B, H, Tq, D) queries -- typically only the *routed* subset of tokens,
        which is the whole point: routing out 75% of tokens makes the scan 4x cheaper.

        Returns
        -------
        y_mem : (B, H, Tq, D) attention over retrieved values; exactly zero for
                streams with an empty bank.
        conf  : (B, Tq) top-1 cosine similarity averaged over heads, in [-1, 1], and
                0 where there was no memory to read. This is the signal that couples
                the memory decision to the depth decision (contribution C1).

        Two-stage by design. The B x H x Tq x M similarity matrix is the largest
        tensor in the model (100M+ entries at M=8192) and keeping it for backward
        would dominate VRAM, so the search runs under no_grad in chunks and only the
        top-k similarities -- B x H x Tq x k, four orders of magnitude smaller -- are
        recomputed with grad so the query still learns.
        """
        B, H, Tq, D = q.shape
        assert B == self.B and H == self.n_head and D == self.head_dim
        k_ret = min(n_neighbors, self.capacity)

        qn = F.normalize(q, dim=-1)
        keys_n = self.keys_norm                                   # maintained on write
        valid = self.valid_mask().view(B, 1, 1, self.capacity)

        y_parts, conf_parts = [], []
        for s in range(0, Tq, chunk_size):
            e = min(s + chunk_size, Tq)
            qc = qn[:, :, s:e]                                    # (B, H, t, D)

            with torch.no_grad():
                sim = torch.matmul(qc.float(), keys_n.transpose(-1, -2))   # (B, H, t, M)
                # -1e4 rather than -inf: an empty bank masks every slot, and a full row
                # of -inf makes softmax emit NaN. A large finite floor stays defined;
                # empty-bank streams are zeroed out below anyway.
                sim = sim.masked_fill(~valid, -1e4)
                top_sim, top_idx = sim.topk(k_ret, dim=-1)        # (B, H, t, k)
                del sim

            # Gather from the pre-normalised keys: no renormalisation needed here.
            k_vec_n = self._flat_gather(self.keys_norm, top_idx).to(qc.dtype)
            v_vec = self._flat_gather(self.values, top_idx)       # (B, H, t, k, D)

            # Recompute similarity from the live query so gradient reaches it.
            sim_live = torch.einsum("bhtd,bhtkd->bhtk", qc, k_vec_n)
            sim_live = sim_live.masked_fill(top_sim < -1e3, -1e4)  # keep padding masked

            attn = F.softmax(sim_live.float() / temperature, dim=-1).to(v_vec.dtype)
            y_parts.append(torch.einsum("bhtk,bhtkd->bhtd", attn, v_vec))
            conf_parts.append(top_sim[..., 0].mean(dim=1))        # (B, t) over heads

        y_mem = torch.cat(y_parts, dim=2)
        conf = torch.cat(conf_parts, dim=1).clamp(-1.0, 1.0)

        # Streams with no memory contribute nothing and report zero confidence.
        live = self.has_memory
        y_mem = torch.where(live.view(B, 1, 1, 1), y_mem, torch.zeros_like(y_mem))
        conf = torch.where(live.view(B, 1), conf, torch.zeros_like(conf))
        return y_mem, conf

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict:
        return {
            "mem_fill_mean": self.fill.float().mean().item(),
            "mem_fill_frac": (self.fill.float() / self.capacity).mean().item(),
            "mem_streams_live": self.has_memory.float().mean().item(),
        }

    def __repr__(self):
        return (f"KVMemoryBank(B={self.B}, heads={self.n_head}, dim={self.head_dim}, "
                f"capacity={self.capacity}, mean_fill={self.fill.float().mean():.0f})")
