# Trittention: Exploring N-way Attention in Transformer Models

This repository contains the implementation and exploration of N-way attention, particularly focusing on 3-way attention (trittention), in transformer models. The goal of this project is to investigate the potential benefits and applications of higher-order attention mechanisms.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Features

- Implementation of standard attention, trittention, trittention cube, local trittention, and mixed attention mechanisms.
- Configurable hyperparameters for customizing the model architecture.
- Preprocessing utilities for handling text data and creating input tensors.
- Evaluation utilities for computing various performance metrics.
- Example scripts for training and evaluating the models on toy problems.
- Comprehensive unit tests for ensuring code correctness and reliability.
- Support for CUDA and Apple MPS for accelerated training on compatible hardware.

## Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To train a model on a toy problem, use the `train_toy_problem.py` script in the `examples` directory:

```
python examples/train_toy_problem.py
```

Modify the script to specify the desired model configuration, dataset paths, and training hyperparameters.

## Code Structure

The code is organized into the following directories:

```
TrittentionTransformer/
├── README.md
├── LICENSE
├── .gitignore
├── MANIFEST.in
├── requirements.txt
├── pyproject.toml
├── data/
│   └── toy_problems/
│       ├── longest_increasing_subsequence.txt
│       └── arithmetic_operations.txt
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── trittention.py
│   ├── trittention_cube.py
│   ├── local_trittention.py
│   └── mixed_attention.py
├── config/
│   ├── __init__.py
│   └── cfgs.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── evaluation_utils.py
├── experiments/
│   ├── induction_head.ipynb [**pending**]
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_trittention.py
│   └── ...
└── examples/
    ├── train_toy_problem.py
    └── ...
```

- `config`: Contains configuration classes for the models.
- `data`: Contains sample datasets for toy problems.
- `examples`: Contains example scripts for training and evaluation.
- `models`: Contains the implementation of various attention mechanisms.
- `tests`: Contains unit tests for the models and utilities.
- `utils`: Contains utility functions for data preprocessing and evaluation.

## Experiments

We conducted experiments on various toy problems to evaluate the performance of different attention mechanisms. The experiments include:

- Longest Increasing Subsequence: Finding the length of the longest increasing subsequence in a given sequence.
- Arithmetic Operations: Evaluating arithmetic expressions and predicting the result.

## Results

Our experiments showed that trittention and its variants (trittention cube, local trittention, mixed attention) outperformed standard attention on certain toy problems, particularly those involving higher-order dependencies and complex patterns.

Detailed results and analysis can be found in the [results](results/) directory.

## Contributing

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to acknowledge the following resources and papers that inspired and influenced this project:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) - Dai et al., 2019
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762). In Advances in neural information processing systems (pp. 5998-6008).
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). [Transformer-xl: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860). arXiv preprint arXiv:1901.02860.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). [Longformer: The long-document transformer](https://arxiv.org/abs/2004.05150). arXiv preprint arXiv:2004.05150.

Feel free to reach out if you have any questions or need further assistance!

---