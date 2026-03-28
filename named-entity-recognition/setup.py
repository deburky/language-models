#!/usr/bin/env python3
"""Set up the brand extractor environment.

Installs Python dependencies and downloads the spaCy English model.
"""

import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a shell command and print success or stderr on failure."""
    print(f"{description}...")
    try:
        _ = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Create folders, install requirements, and download the spaCy model."""
    print("=== Brand Extractor Setup ===")

    # Create necessary directories
    print("Creating directories...")
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("scripts").mkdir(exist_ok=True)

    # Install Python dependencies
    if not run_command(
        "pip install -r requirements.txt", "Installing Python dependencies"
    ):
        print(
            "Failed to install dependencies. Please check your Python environment."
        )
        return False

    # Download spaCy model
    if not run_command(
        "python -m spacy download en_core_web_sm", "Downloading spaCy English model"
    ):
        print(
            "Failed to download spaCy model. You can still use pattern-based extraction."
        )

    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Place your CSV file in the 'data' directory")
    print("2. Update the CSV_IN path in scripts/brand_extractor.py if needed")
    print("3. Run: python scripts/brand_extractor.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
