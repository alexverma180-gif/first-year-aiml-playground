import pandas as pd
import sys
import os
import pytest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Add the project's parent directory to the Python path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import load_data, train_model

def test_load_data():
    """Tests that load_data() returns a non-empty DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_train_model():
    """Tests that train_model() returns a trained model."""
    df = load_data()
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, k=5)
    assert isinstance(model, KNeighborsClassifier)
