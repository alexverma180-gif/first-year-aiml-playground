import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from app.model_utils import load_data, train_model
from scratch_ml.linear_regression import LinearRegressionGD

@pytest.fixture(scope="module")
def app_data():
    """Fixture to load data once for all performance tests."""
    # Use the __wrapped__ attribute to bypass the Streamlit caching decorator
    df = load_data.__wrapped__()
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def test_load_data_performance(benchmark):
    """Benchmark the load_data function."""
    # Use the __wrapped__ attribute to bypass the Streamlit caching decorator
    benchmark(load_data.__wrapped__)

def test_train_model_performance(benchmark, app_data):
    """Benchmark the train_model function."""
    X_train, _, y_train, _ = app_data
    # Use the __wrapped__ attribute to bypass the Streamlit caching decorator
    benchmark(train_model.__wrapped__, X_train, y_train, k=5)

@pytest.fixture(scope="module")
def linear_regression_data():
    """Fixture to generate a large dataset for performance tests."""
    rng = np.random.default_rng(42)
    num_samples = 10000
    X = rng.uniform(-10, 10, size=(num_samples, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + 7 + rng.normal(0, 0.5, size=num_samples)
    return X, y

def test_linear_regression_performance(benchmark, linear_regression_data):
    """Benchmark the LinearRegressionGD model."""
    X, y = linear_regression_data
    model = LinearRegressionGD(lr=0.01, epochs=1000)
    benchmark(model.fit, X, y)


@pytest.fixture(scope="module")
def trained_model(app_data):
    """Fixture to train a model once for all prediction tests."""
    X_train, _, y_train, _ = app_data
    model = train_model.__wrapped__(X_train, y_train, k=5)
    return model


def test_prediction_performance(benchmark, trained_model, app_data):
    """Benchmark the model's predict method."""
    _, X_test, _, _ = app_data
    benchmark(trained_model.predict, X_test)
