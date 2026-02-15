import numpy as np
import time
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
        two_over_n = 2.0 / n

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            grad_w = two_over_n * (X.T @ error)
            grad_b = two_over_n * np.sum(error)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

class AugmentedLinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        # Augment X with a column of ones
        X_aug = np.column_stack([X, np.ones(n)])
        self.theta = np.zeros(d + 1)

        # Precompute XTX and XTy
        XTX = X_aug.T @ X_aug
        XTy = X_aug.T @ y

        learning_rate_factor = self.lr * (2.0 / n)
        XTX_scaled = learning_rate_factor * XTX
        XTy_scaled = learning_rate_factor * XTy

        for _ in range(self.epochs):
            step = XTX_scaled @ self.theta - XTy_scaled
            self.theta -= step
        return self

def benchmark_models():
    n = 10000
    d = 20
    epochs = 1000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"Benchmarking with N={n}, D={d}, Epochs={epochs}")

    # Naive
    model_naive = NaiveLinearRegressionGD(epochs=epochs)
    start = time.perf_counter()
    model_naive.fit(X, y)
    end = time.perf_counter()
    print(f"Naive GD: {end - start:.4f}s")

    # Optimized (Current implementation)
    model_opt = LinearRegressionGD(epochs=epochs)
    start = time.perf_counter()
    model_opt.fit(X, y)
    end = time.perf_counter()
    print(f"Optimized GD: {end - start:.4f}s")

    # Augmented
    model_aug = AugmentedLinearRegressionGD(epochs=epochs)
    start = time.perf_counter()
    model_aug.fit(X, y)
    end = time.perf_counter()
    print(f"Augmented GD: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_models()
