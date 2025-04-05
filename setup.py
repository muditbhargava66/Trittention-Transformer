"""
Setup script for the Trittention-Transformer project.
"""

import os
from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="trittention-transformer",
    version="0.2.0",
    description="Exploring N-way Attention in Transformer Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mudit Bhargava",
    author_email="muditbhargava666@gmail.com",  # Replace with actual email
    url="https://github.com/muditbhargava66/Trittention-Transformer",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="transformer, attention, nlp, deep-learning, machine-learning",
    python_requires=">=3.10",
)
