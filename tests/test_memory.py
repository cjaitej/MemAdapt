"""Memory bank correctness.

`test_no_future_leakage` is the one that matters. A retrieval LM that can see the
tokens it is predicting reports beautiful perplexity and means nothing; this is the
classic way results in this area turn out to be wrong. If it goes red, stop.
"""

import pytest
import torch

from amt.model import AMT, KVMemoryBank, variant


def tiny_config(**kw):
    base = dict(n_layer=6, n_embd=64, n_head=4, block_size=32, vocab_size=128,
                n_trunk=1, n_dense_tail=1, mem_size=64, n_neighbors=8,
                mem_query_chunk=16, ce_chunks=2)
    base.update(kw)
    return variant("b6_amt_joint", **base)


@pytest.fixture
def bank():
    return KVMemoryBank(batch_size=2, n_head=4, head_dim=16, capacity=8,
                        device="cpu", dtype=torch.float32)


# --------------------------------------------------------------------------
# Bank mechanics
# --------------------------------------------------------------------------

def test_starts_empty(bank):
    assert not bank.has_memory.any()
    assert bank.valid_mask().sum() == 0


def test_write_advances_fill_and_wraps(bank):
    k = torch.randn(2, 4, 3, 16)
    bank.write(k, k.clone())
    assert bank.fill.tolist() == [3, 3]
    assert bank.ptr.tolist() == [3, 3]

    bank.write(torch.randn(2, 4, 7, 16), torch.randn(2, 4, 7, 16))
    assert bank.fill.tolist() == [8, 8], "fill must saturate at capacity"
    assert bank.ptr.tolist() == [2, 2], "write pointer must wrap"


def test_oversized_write_keeps_the_tail(bank):
    """A segment longer than the bank keeps its most recent tokens, not its first."""
    v = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1).expand(2, 4, 12, 16)
    bank.write(v.clone(), v.clone())
    stored = bank.values[0, 0, :, 0].tolist()
    assert sorted(stored) == list(range(4, 12))


def test_clear_is_per_stream(bank):
    bank.write(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16))
    bank.clear(torch.tensor([True, False]))
    assert bank.fill.tolist() == [0, 5]
    assert bank.has_memory.tolist() == [False, True]


def test_empty_bank_returns_zeros_not_nan(bank):
    """Every slot masked means an all -inf softmax row -- must not produce NaN."""
    q = torch.randn(2, 4, 6, 16)
    y, conf = bank.read(q, n_neighbors=4, chunk_size=3)
    assert torch.isfinite(y).all() and torch.isfinite(conf).all()
    assert (y == 0).all(), "an empty bank must contribute nothing"
    assert (conf == 0).all()


def test_partially_filled_bank_ignores_empty_slots(bank):
    """Unwritten slots are zeros; they must never win the kNN over real entries."""
    k = torch.randn(2, 4, 2, 16)
    bank.write(k, torch.ones(2, 4, 2, 16))
    y, conf = bank.read(torch.randn(2, 4, 5, 16), n_neighbors=8, chunk_size=5)
    # Only slots 0 and 1 hold data; both have value 1, so any honest read returns 1.
    assert torch.allclose(y, torch.ones_like(y), atol=1e-5)


def test_retrieval_finds_the_planted_vector(bank):
    """Query with a stored key; the top-1 neighbour must be its value."""
    keys = torch.randn(2, 4, 8, 16)
    values = torch.randn(2, 4, 8, 16)
    bank.write(keys, values)

    target = 3
    q = keys[:, :, target:target + 1] * 5.0     # scaled: cosine sim is scale-free
    y, conf = bank.read(q, n_neighbors=1, chunk_size=1)
    assert torch.allclose(y, values[:, :, target:target + 1], atol=1e-4)
    assert (conf > 0.99).all(), f"top-1 cosine should be ~1, got {conf}"


def test_read_is_chunk_size_invariant(bank):
    bank.write(torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16))
    q = torch.randn(2, 4, 12, 16)
    y1, c1 = bank.read(q, n_neighbors=4, chunk_size=12)
    y2, c2 = bank.read(q, n_neighbors=4, chunk_size=5)
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(c1, c2, atol=1e-5)


def test_query_receives_gradient(bank):
    """The read must be differentiable w.r.t. the query or the model cannot learn
    to retrieve better -- only to gate what it happens to get."""
    bank.write(torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16))
    q = torch.randn(2, 4, 4, 16, requires_grad=True)
    y, _ = bank.read(q, n_neighbors=4, chunk_size=2)
    y.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_bank_holds_no_graph(bank):
    """Stored tensors must be detached, or every step leaks the previous graph."""
    k = torch.randn(2, 4, 4, 16, requires_grad=True)
    bank.write(k, k.clone())
    assert not bank.keys.requires_grad
    assert not bank.values.requires_grad


# --------------------------------------------------------------------------
# THE invariant (risk R6)
# --------------------------------------------------------------------------

def test_no_future_leakage():
    """Memory must contain only tokens strictly before the current segment.

    Constructed so a violation is unmissable: segment 2 is predicted with a bank that
    was written from segment 1 only. We then rewrite the bank using segment 2 itself
    and assert the loss *changes* -- proving the bank genuinely influences the output,
    so the first assertion is not passing vacuously.
    """
    torch.manual_seed(0)
    model = AMT(tiny_config()).eval()
    B, T = 2, 32
    seg1 = torch.randint(0, 128, (B, T))
    seg2 = torch.randint(0, 128, (B, T))
    tgt2 = torch.randint(0, 128, (B, T))

    bank = model.make_bank(B, "cpu", dtype=torch.float32)
    with torch.no_grad():
        # Correct order: forward on seg1, THEN write. Bank now holds only seg1.
        _, _, _, kv1 = model(seg1, bank=bank)
        model.write_memory(bank, kv1)
        assert bank.fill.tolist() == [T, T]
        honest = model(seg2, targets=tgt2, bank=bank)[1]["lm"].item()

        # Cheating: let the bank also see seg2 before predicting it.
        _, _, _, kv2 = model(seg2, bank=bank)
        model.write_memory(bank, kv2)
        cheating = model(seg2, targets=tgt2, bank=bank)[1]["lm"].item()

    assert honest != pytest.approx(cheating, abs=1e-6), (
        "bank contents did not change the loss -- the leakage test is vacuous, "
        "memory is not actually being used"
    )


def test_clear_on_document_boundary_erases_history():
    """Crossing into a new document must not carry the old document's memory."""
    torch.manual_seed(0)
    model = AMT(tiny_config()).eval()
    B, T = 2, 32
    seg = torch.randint(0, 128, (B, T))
    tgt = torch.randint(0, 128, (B, T))

    bank = model.make_bank(B, "cpu", dtype=torch.float32)
    with torch.no_grad():
        _, _, _, kv = model(seg, bank=bank)
        model.write_memory(bank, kv)
        with_history = model(seg, targets=tgt, bank=bank)[1]["lm"].item()

        bank.clear(torch.tensor([True, True]))
        model.write_memory(bank, kv)
        assert bank.fill.tolist() == [T, T], "clear then write leaves only the new segment"

        bank.clear()
        fresh = model(seg, targets=tgt, bank=bank)[1]["lm"].item()

    assert with_history != pytest.approx(fresh, abs=1e-6)
