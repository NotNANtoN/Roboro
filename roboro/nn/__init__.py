"""Shared neural-network building blocks.

``MLPBlock`` is the standard trunk used by encoders, critics, and actors.
Swap it for any drop-in replacement (SambaV2, Mamba-MLP, …) to change the
backbone everywhere at once.
"""

from roboro.nn.blocks import MLPBlock, get_activation
