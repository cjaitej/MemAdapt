"""AdaptiveGPT.

A GPT that learns, per token, how many transformer layers to spend on it. Easy tokens
exit early and are unembedded from wherever they stopped; hard tokens run the full
stack.

Built on Karpathy's nanoGPT.
"""

__version__ = "0.2.0"
