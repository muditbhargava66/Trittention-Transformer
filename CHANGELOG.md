# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-04-05

### Added
- Optimized implementations of trittention:
  - SparseTrittention with pruning for efficiency
  - WindowedTrittention for better scaling with long sequences
- PyTorch Lightning integration for modern training workflows
- Comprehensive utility modules:
  - Data loading and processing utilities
  - Evaluation and benchmarking tools
  - Attention visualization utilities
- New example applications:
  - Text classification with trittention
  - Sequence modeling with different attention mechanisms
- Full documentation:
  - API reference
  - Usage examples
  - Performance analysis
- Benchmarking and model comparison scripts
- Hyperparameter tuning support
- Model conversion between attention types

### Changed
- Updated project structure for better organization
- Modernized codebase with Python 3.10+ features
- Enhanced README with clearer explanations and examples
- Improved type hinting throughout the codebase
- Better error handling and assertions

### Fixed
- Memory leaks in attention computation
- Numerical stability in trittention calculations
- Edge cases in sequence handling
- Fixed dtype mismatch in model's forward pass
- Added support for token embedding for sequence data
- Fixed compatibility issues between different types of datasets
- Resolved shape mismatches in hyperparameter tuning
- Improved model architecture with enhanced TrittentionSequenceModel that supports both feature-based and token-based inputs
- Fixed input/output size determination for different dataset types
- Added backward compatibility for existing code using the previous model interface

## [0.1.0] - 2023-08-15

### Added
- Initial implementation of attention mechanisms:
  - Standard attention
  - Trittention (3-way attention)
  - Trittention cube
  - Local trittention
  - Mixed attention
- Basic evaluation scripts for toy problems
- Sample datasets for testing
- Project structure and configuration