import pytest
import numpy as np
from scratch_ml.linear_regression import LinearRegressionGD

class NaiveLinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            # Standard gradient descent updates O(N*D)
            dw = (2.0 / n) * (X.T @ error)
            db = (2.0 / n) * np.sum(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

class AugmentedLinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w_aug = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        # Augmented matrix [X, 1] to handle bias within the weight vector
        X_aug = np.column_stack((X, np.ones(n)))
        self.w_aug = np.zeros(d + 1)

        # Precompute XTX and XTy for O(D^2) complexity per iteration
        XTX = X_aug.T @ X_aug
        XTy = X_aug.T @ y

        learning_rate_factor = self.lr * (2.0 / n)
        XTX_scaled = learning_rate_factor * XTX
        XTy_scaled = learning_rate_factor * XTy

        for _ in range(self.epochs):
            # Single matrix-vector multiplication per iteration
            grad = XTX_scaled @ self.w_aug - XTy_scaled
            self.w_aug -= grad
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_aug = np.column_stack((X, np.ones(X.shape[0])))
        return X_aug @ self.w_aug

@pytest.fixture
def large_dataset():
    """Generate a large synthetic dataset for comparison benchmarks."""
    rng = np.random.default_rng(42)
    n, d = 10000, 20
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    return X, y

@pytest.mark.benchmark(group="comparison")
def test_naive_performance(benchmark, large_dataset):
    """Benchmark the naive implementation."""
    X, y = large_dataset
    model = NaiveLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=20)

@pytest.mark.benchmark(group="comparison")
def test_optimized_performance(benchmark, large_dataset):
    """Benchmark the current optimized implementation."""
    X, y = large_dataset
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=20)

@pytest.mark.benchmark(group="comparison")
def test_augmented_performance(benchmark, large_dataset):
    """Benchmark the augmented matrix optimized implementation."""
    X, y = large_dataset
    model = AugmentedLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=20)
