import pandas as pd
import sys
import os
import pytest

# Add the project's parent directory to the Python path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import load_data

def test_load_data():
    """Tests that load_data() returns a non-empty DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
