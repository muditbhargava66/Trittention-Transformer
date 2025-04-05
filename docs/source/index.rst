Welcome to Trittention-Transformer's documentation!
===========================================

.. image:: https://github.com/muditbhargava66/Trittention-Transformer/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/muditbhargava66/Trittention-Transformer/actions/workflows/ci.yml
   :alt: CI

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :alt: Python

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Trittention-Transformer is a project exploring N-way attention mechanisms in transformer models, with a focus on 3-way attention (trittention).

Unlike standard self-attention which captures pairwise interactions, trittention captures higher-order relationships between tokens, potentially enabling more expressive modeling of complex dependencies.

Key Features
-----------

* Multiple attention mechanism implementations (standard, trittention, sparse variants)
* Optimized sparse and windowed implementations for computational efficiency 
* Comprehensive PyTorch and PyTorch Lightning integrations
* Visualization tools for attention patterns
* Benchmarking utilities for complexity analysis

Getting Started
--------------

Installation:

.. code-block:: bash

   pip install -r requirements.txt

Quick Start:

.. code-block:: python
   
   import torch
   from config.cfgs import TrittentionConfig
   from models.trittention import Trittention
   
   # Create configuration
   config = TrittentionConfig(
       hidden_size=64,
       num_attention_heads=4,
       attention_probs_dropout_prob=0.1
   )
   
   # Initialize model
   model = Trittention(config)
   
   # Generate sample input
   batch_size, seq_length = 2, 10
   hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
   
   # Forward pass
   output = model(hidden_states)
   print(output.shape)  # [2, 10, 64]

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   introduction
   installation
   attention_mechanisms
   usage_examples
   performance
   api_reference/index
   contributing
   faq

Attribution
-----------

If you use this code in your research, please cite our work:

.. code-block:: text

   @software{bhargava2023trittention,
     author = {Bhargava, Mudit},
     title = {Trittention-Transformer: Exploring N-way Attention in Transformer Models},
     year = {2023},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\\url{https://github.com/muditbhargava66/Trittention-Transformer}}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
