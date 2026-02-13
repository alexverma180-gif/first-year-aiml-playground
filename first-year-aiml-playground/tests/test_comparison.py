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
            grad_w = (2.0 / n) * (X.T @ error)
            grad_b = (2.0 / n) * np.sum(error)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

class AugmentedLinearRegressionGD:
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

        XTX = X.T @ X
        XTy = X.T @ y
        X_sum = X.sum(axis=0)
        y_sum = y.sum()

        learning_rate_factor = self.lr * (2.0 / n)

        A_scaled = np.zeros((d + 1, d + 1))
        A_scaled[:d, :d] = learning_rate_factor * XTX
        A_scaled[:d, d] = learning_rate_factor * X_sum
        A_scaled[d, :d] = learning_rate_factor * X_sum
        A_scaled[d, d] = learning_rate_factor * n

        b_scaled_aug = np.concatenate([learning_rate_factor * XTy, [learning_rate_factor * y_sum]])

        theta = np.concatenate([self.w, [self.b]])
        for _ in range(self.epochs):
            theta -= (A_scaled @ theta - b_scaled_aug)

        self.w = theta[:-1]
        self.b = theta[-1]
        return self

@pytest.fixture(scope="module")
def large_dataset():
    rng = np.random.default_rng(42)
    n, d = 10000, 100
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    return X, y

def test_naive_vs_optimized_correctness(large_dataset):
    X, y = large_dataset
    epochs = 100
    lr = 0.01

    naive_model = NaiveLinearRegressionGD(lr=lr, epochs=epochs).fit(X, y)
    optimized_model = LinearRegressionGD(lr=lr, epochs=epochs).fit(X, y)

    np.testing.assert_allclose(naive_model.w, optimized_model.w, rtol=1e-5)
    np.testing.assert_allclose(naive_model.b, optimized_model.b, rtol=1e-5)

def test_naive_performance(benchmark, large_dataset):
    X, y = large_dataset
    model = NaiveLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

def test_optimized_performance(benchmark, large_dataset):
    X, y = large_dataset
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

def test_augmented_performance(benchmark, large_dataset):
    X, y = large_dataset
    model = AugmentedLinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)
