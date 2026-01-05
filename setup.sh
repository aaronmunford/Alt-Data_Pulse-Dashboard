#!/bin/bash

# Alt-Data Pulse Dashboard Setup Script
# This script sets up the Python virtual environment and installs dependencies

set -e  # Exit on error

echo "======================================"
echo "Alt-Data Pulse Dashboard Setup"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)

echo "Python version: $PYTHON_VERSION"

# Validate Python version (requires 3.8+)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. You have Python $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dashboard dependencies from requirements.txt
echo "Installing dashboard dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "✓ Setup complete!"
echo "======================================"
echo ""
echo "To run the dashboard:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Navigate to dashboard directory: cd dashboard"
echo "  3. Run the app: streamlit run app.py"
echo ""
echo "Or run this one-liner:"
echo "  source .venv/bin/activate && cd dashboard && streamlit run app.py"
echo ""
