"""
Initialization script for the Trittention-Transformer project.

This script sets up the necessary directories and files for the project.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse


def create_directory(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")


def initialize_project(base_dir, force=False):
    """
    Initialize the project directory structure.
    
    Args:
        base_dir: Base directory of the project
        force: Whether to force reinitialization
    """
    base_dir = Path(base_dir)
    
    # Check if directories already exist
    existing_dirs = [d for d in [
        base_dir / "results",
        base_dir / "results" / "logs",
        base_dir / "results" / "checkpoints",
        base_dir / "results" / "visualizations"
    ] if d.exists()]
    
    if existing_dirs and not force:
        print("Some directories already exist. Use --force to reinitialize.")
        return
    
    # Create necessary directories
    create_directory(base_dir / "results")
    create_directory(base_dir / "results" / "logs")
    create_directory(base_dir / "results" / "checkpoints")
    create_directory(base_dir / "results" / "visualizations")
    
    # Create .gitkeep files to ensure directories are tracked by git
    for directory in [
        base_dir / "results" / "logs",
        base_dir / "results" / "checkpoints",
        base_dir / "results" / "visualizations"
    ]:
        gitkeep_file = directory / ".gitkeep"
        with open(gitkeep_file, "w") as f:
            pass
        print(f"Created file: {gitkeep_file}")
    
    print("Project initialized successfully!")


def check_python_version():
    """Check that the Python version is at least 3.10."""
    if sys.version_info < (3, 10):
        print("Error: This project requires Python 3.10 or higher.")
        sys.exit(1)


def check_dependencies():
    """Check that the required dependencies are installed."""
    try:
        import torch
        import numpy
        import matplotlib
        import pytorch_lightning
        print("All core dependencies found!")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all dependencies with: pip install -r requirements.txt")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Initialize the Trittention-Transformer project")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory of the project")
    parser.add_argument("--force", action="store_true",
                        help="Force reinitialization even if directories exist")
    parser.add_argument("--check_only", action="store_true",
                        help="Only check dependencies without initializing")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Check Python version
    check_python_version()
    
    # Check dependencies
    check_dependencies()
    
    if not args.check_only:
        # Initialize project
        initialize_project(args.base_dir, args.force)


if __name__ == "__main__":
    main()
