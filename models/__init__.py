"""
Models module for Trittention-Transformer project.

This module contains implementations of various attention mechanisms
including standard attention, trittention, and optimized variants.
"""

from models.attention import Attention
from models.trittention import Trittention
from models.trittention_cube import TrittentionCube
from models.local_trittention import LocalTrittention
from models.mixed_attention import MixedAttention
from models.sparse_trittention import SparseTrittention, WindowedTrittention
from models.lightning_module import (
    TrittentionSequenceModel,
    TrittentionLightningModule,
    TrittentionDataModule
)

__all__ = [
    # Attention mechanisms
    'Attention',
    'Trittention',
    'TrittentionCube',
    'LocalTrittention',
    'MixedAttention',
    'SparseTrittention',
    'WindowedTrittention',
    
    # PyTorch Lightning modules
    'TrittentionSequenceModel',
    'TrittentionLightningModule',
    'TrittentionDataModule'
]
