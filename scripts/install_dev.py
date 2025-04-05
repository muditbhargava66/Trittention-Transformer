"""
Install the Trittention-Transformer package in development mode.

This script installs the package in development mode, which makes it easier to
modify the code and see the changes without reinstalling the package.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Install the package in development mode."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    print(f"Installing from {project_dir}")
    
    # Run pip install in development mode
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    try:
        subprocess.run(cmd, cwd=project_dir, check=True)
        print("\nSuccessfully installed in development mode!")
        print("You can now import the package as 'import models', 'import utils', etc.")
    except subprocess.CalledProcessError as e:
        print(f"\nError installing package: {e}")
        sys.exit(1)
    
    # Test the installation
    print("\nTesting installation...")
    try:
        # Test imports
        import_test_cmd = [
            sys.executable, 
            "-c", 
            "import config; import models; import utils; print('Import test successful!')"
        ]
        subprocess.run(import_test_cmd, cwd=project_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError testing installation: {e}")
        sys.exit(1)
    
    print("\nAll done! You can now use the package in development mode.")

if __name__ == "__main__":
    main()
