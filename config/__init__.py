"""
Configuration module for the Trittention-Transformer project.

This module provides configuration classes and utilities for configuring
different attention mechanisms and models.
"""

from config.cfgs import TrittentionConfig
from config.sample_config import (
    get_default_config,
    get_standard_attention_config,
    get_local_trittention_config,
    get_mixed_attention_config,
    get_sparse_trittention_config,
    get_long_sequence_config,
    get_small_model_config,
    get_lowrank_config
)

__all__ = [
    'TrittentionConfig',
    'get_default_config',
    'get_standard_attention_config',
    'get_local_trittention_config',
    'get_mixed_attention_config',
    'get_sparse_trittention_config',
    'get_long_sequence_config',
    'get_small_model_config',
    'get_lowrank_config'
]
