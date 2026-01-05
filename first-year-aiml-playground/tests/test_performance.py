import pytest
from sklearn.model_selection import train_test_split
from app.model_utils import load_data, train_model

@pytest.fixture(scope="module")
def app_data():
    """Fixture to load data once for all performance tests."""
    df = load_data()
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def test_load_data_performance(benchmark):
    """Benchmark the load_data function."""
    benchmark(load_data)

def test_train_model_performance(benchmark, app_data):
    """Benchmark the train_model function."""
    X_train, _, y_train, _ = app_data
    benchmark(train_model, X_train, y_train, k=5)
