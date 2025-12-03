#!/bin/bash
# Quick script to regenerate citation half-life graph

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Regenerating citation half-life analysis graph..."
echo ""

# Try to run with python3
if command -v python3 &> /dev/null; then
    python3 regenerate_citation_half_life.py
else
    echo "Error: python3 not found. Please ensure Python 3 is installed."
    exit 1
fi
