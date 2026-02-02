#!/bin/bash
# Script to run performance benchmarks and show results

echo "Running performance benchmarks..."
PYTHONPATH=./first-year-aiml-playground python3 -m pytest first-year-aiml-playground/tests/test_performance.py --benchmark-only
