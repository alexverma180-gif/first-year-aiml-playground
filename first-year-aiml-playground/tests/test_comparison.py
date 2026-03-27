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
            # Naive O(ND) gradient calculation
            y_pred = X @ self.w + self.b
            error = y_pred - y
            dw = (2.0/n) * (X.T @ error)
            db = (2.0/n) * np.sum(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

def test_naive_vs_optimized_equivalence():
    """Verify that both implementations produce similar results."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = X @ rng.standard_normal(5) + rng.standard_normal(100)

    lr = 0.01
    epochs = 100

    model_opt = LinearRegressionGD(lr=lr, epochs=epochs).fit(X, y)
    model_naive = NaiveLinearRegressionGD(lr=lr, epochs=epochs).fit(X, y)

    np.testing.assert_allclose(model_opt.w, model_naive.w, rtol=1e-5)
    np.testing.assert_allclose(model_opt.b, model_naive.b, rtol=1e-5)

def test_benchmark_optimized(benchmark):
    """Benchmark the optimized implementation."""
    n, d = 10000, 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

def test_benchmark_naive(benchmark):
    """Benchmark the naive implementation."""
    n, d = 10000, 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    model = NaiveLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

def test_benchmark_optimized_large_n(benchmark):
    """Benchmark the optimized implementation with very large N."""
    n, d = 100000, 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

def test_benchmark_naive_large_n(benchmark):
    """Benchmark the naive implementation with very large N."""
    n, d = 100000, 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    model = NaiveLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)
