import numpy as np
import time
import resource
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath("first-year-aiml-playground"))

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
            # Naive gradient calculation: O(N*D) per iteration
            y_pred = X @ self.w + self.b
            error = y_pred - y

            grad_w = (2.0 / n) * (X.T @ error)
            grad_b = (2.0 / n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_memory_usage():
    # Returns peak RSS in MB
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return usage / (1024 * 1024)
    else:
        return usage / 1024

def profile_comparison():
    n = 10000
    d = 10
    epochs = 1000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"--- Comparison: Optimized vs Naive GD (N={n}, D={d}, Epochs={epochs}) ---")

    # Optimized
    start_time = time.perf_counter()
    LinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    opt_time = time.perf_counter() - start_time
    print(f"Optimized GD: {opt_time:.4f} seconds")

    # Naive
    start_time = time.perf_counter()
    NaiveLinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    naive_time = time.perf_counter() - start_time
    print(f"Naive GD:     {naive_time:.4f} seconds")
    print(f"Speedup:      {naive_time / opt_time:.2f}x")

def profile_high_dim():
    n = 10000
    d = 500
    epochs = 100
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"\n--- High-Dimensional Analysis (N={n}, D={d}, Epochs={epochs}) ---")

    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    LinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    end_time = time.perf_counter() - start_time
    mem_after = get_memory_usage()

    print(f"Training time: {end_time:.4f} seconds")
    print(f"Peak Memory:   {mem_after:.2f} MB")
    print(f"Memory Delta:  {mem_after - mem_before:.2f} MB (approximate)")

if __name__ == "__main__":
    profile_comparison()
    profile_high_dim()
