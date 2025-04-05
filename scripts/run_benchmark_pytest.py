#!/usr/bin/env python3
"""
Pytest-compatible benchmark runner.

This script uses a pytest environment variable to trigger the benchmark test.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    # Set environment variable for test
    os.environ["PYTEST_BENCHMARK"] = "1"
    
    print("Running benchmark via pytest...")
    
    # Run pytest with the specific test
    cmd = [
        "pytest", 
        "tests/test_sparse_trittention.py::TestSparseTrittention::test_benchmark",
        "-v"
    ]
    
    # Execute the command
    subprocess.run(cmd, check=True)
