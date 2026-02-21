import numpy as np
import time
import resource
import sys
import os

# Add the project root to sys.path to allow imports from scratch_ml
sys.path.append(os.path.abspath("first-year-aiml-playground"))

try:
    from scratch_ml.linear_regression import LinearRegressionGD
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.path.abspath("."))
    from scratch_ml.linear_regression import LinearRegressionGD

class NaiveLinearRegressionGD:
    """A naive implementation of Linear Regression with Gradient Descent for performance comparison."""
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

            # Standard gradient descent updates: O(N*D) per iteration
            grad_w = (2/n) * (X.T @ error)
            grad_b = (2/n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def profile_performance():
    print("--- Performance Comparison: Optimized vs Naive ---")
    N = 100000
    D = 10
    epochs = 100

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D))
    y = X @ rng.standard_normal(D) + rng.standard_normal(N)

    # Naive version
    naive_model = NaiveLinearRegressionGD(lr=0.01, epochs=epochs)
    start = time.perf_counter()
    naive_model.fit(X, y)
    naive_time = time.perf_counter() - start
    print(f"Naive GD Fit (N={N}, D={D}, epochs={epochs}): {naive_time:.4f}s")

    # Optimized version
    opt_model = LinearRegressionGD(lr=0.01, epochs=epochs)
    start = time.perf_counter()
    opt_model.fit(X, y)
    opt_time = time.perf_counter() - start
    print(f"Optimized GD Fit (N={N}, D={D}, epochs={epochs}): {opt_time:.4f}s")

    speedup = naive_time / opt_time
    print(f"Speedup: {speedup:.2f}x")

def profile_scaling():
    print("\n--- Scaling Analysis with D (N=10000, epochs=100) ---")
    N = 10000
    epochs = 100
    for D in [5, 50, 500]:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((N, D))
        y = X @ rng.standard_normal(D) + rng.standard_normal(N)

        model = LinearRegressionGD(lr=0.01, epochs=epochs)
        start = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - start
        print(f"D={D:3}: {elapsed*1000:7.2f} ms")

def profile_memory():
    print("\n--- Memory Usage Analysis (High-D) ---")
    N = 10000
    D = 500
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D))
    y = X @ rng.standard_normal(D) + rng.standard_normal(N)

    model = LinearRegressionGD(lr=0.01, epochs=1000)
    # Start tracking memory before fit
    model.fit(X, y)

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux, ru_maxrss is in kilobytes
    print(f"Peak Memory Usage for N={N}, D={D}: {usage / 1024:.2f} MB")

if __name__ == "__main__":
    profile_performance()
    profile_scaling()
    profile_memory()
