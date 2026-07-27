"""Adaptive Memory Transformer (AMT).

A GPT that learns, per token, how much computation (depth) and how much external
memory (kNN retrieval over its own KV cache) to spend.

Built on Karpathy's nanoGPT (`train_gpt2.py` at the repo root, kept untouched as
the reference implementation).
"""

__version__ = "0.1.0"
