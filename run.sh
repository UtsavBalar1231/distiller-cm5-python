#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage: ./run.sh [options]"
    echo "Options:"
    echo "  --gui    Launch the GUI interface instead of CLI"
    echo "  -h, --help    Display this help message"
    echo ""
    echo "All other options are passed directly to main.py"
}

# Check for help flag
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

# Check if the .venv directory exists
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Installing dependencies..."
  # Install dependencies using uv
  uv init
  uv venv
  uv pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
  fi
else
  echo "Virtual environment found."
fi

# Activate the virtual environment
source .venv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate virtual environment."
  exit 1
fi

# Run the main Python script
echo "Running main.py..."
python main.py "$@"

# Deactivate the virtual environment (optional, runs when script exits)
# deactivate 
