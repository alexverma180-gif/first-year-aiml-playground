
import pytest
from sklearn.model_selection import train_test_split
from app.model_utils import load_data, train_model

@pytest.fixture(scope="module")
def app_data():
    """Fixture to load and split data for performance tests."""
    df = load_data.__wrapped__()
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, y_train

def test_load_data_performance(benchmark):
    """Benchmark the load_data function."""
    benchmark(load_data.__wrapped__)

def test_train_model_performance(benchmark, app_data):
    """Benchmark the train_model function."""
    X_train, y_train = app_data
    k = 5
    benchmark.pedantic(train_model.__wrapped__, args=(X_train, y_train, k), rounds=10, iterations=5)
