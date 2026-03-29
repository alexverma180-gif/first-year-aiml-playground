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
    benchmark.pedantic(train_model.__wrapped__, args=(X_train, y_train), kwargs={"k": 5}, rounds=10)

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
    benchmark.pedantic(model.fit, args=(X, y), rounds=10)


def test_linear_regression_naive_performance(benchmark, linear_regression_data):
    """Benchmark a naive implementation of Linear Regression GD (no precomputation)."""
    X, y = linear_regression_data
    lr = 0.01
    epochs = 1000

    def naive_fit(X, y, lr, epochs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        two_over_n = 2.0 / n
        learning_rate_factor = lr * two_over_n
        for _ in range(epochs):
            # Naive gradient calculation: O(N*D) per epoch
            error = (X @ w + b) - y
            grad_w = X.T @ error
            grad_b = np.sum(error)
            w -= learning_rate_factor * grad_w
            b -= learning_rate_factor * grad_b
        return w, b

    benchmark.pedantic(naive_fit, args=(X, y, lr, epochs), rounds=10)


@pytest.fixture(scope="module")
def trained_linear_regression_model(linear_regression_data):
    """Fixture to train a LinearRegressionGD model once for all prediction tests."""
    X, y = linear_regression_data
    model = LinearRegressionGD(lr=0.01, epochs=1000)
    model.fit(X, y)
    return model


def test_linear_regression_predict_performance(benchmark, trained_linear_regression_model, linear_regression_data):
    """Benchmark the predict method of the LinearRegressionGD model."""
    X, _ = linear_regression_data
    benchmark.pedantic(trained_linear_regression_model.predict, args=(X,), rounds=10)


@pytest.fixture(scope="module")
def trained_model(app_data):
    """Fixture to train a model once for all prediction tests."""
    X_train, _, y_train, _ = app_data
    # Use the __wrapped__ attribute to bypass the Streamlit caching decorator
    return train_model.__wrapped__(X_train, y_train, k=5)

def test_predict_model_performance(benchmark, trained_model, app_data):
    """Benchmark the predict method of the trained model."""
    _, X_test, _, _ = app_data
    benchmark.pedantic(trained_model.predict, args=(X_test,), rounds=10)
