import numpy as np
import time
import resource
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scratch_ml.linear_regression import LinearRegressionGD

class LinearRegressionGDNaive:
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
            # Naive gradient descent: O(N*D) per iteration
            y_pred = X @ self.w + self.b
            error = y_pred - y

            grad_w = (2/n) * (X.T @ error)
            grad_b = (2/n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_peak_memory():
    """Returns peak memory usage in MB."""
    # On Linux, ru_maxrss is in kilobytes.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0

def profile_performance(N, D, epochs=100):
    print(f"\n--- Profiling with N={N:7,}, D={D:3}, Epochs={epochs} ---")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D))
    y = X @ rng.standard_normal(D) + 2.0 + rng.standard_normal(N) * 0.1

    # Naive version
    model_naive = LinearRegressionGDNaive(lr=0.01, epochs=epochs)
    start_time = time.perf_counter()
    model_naive.fit(X, y)
    end_time = time.perf_counter()
    naive_duration = end_time - start_time
    print(f"Naive Fit Time:     {naive_duration:10.4f}s")

    # Optimized version
    model_opt = LinearRegressionGD(lr=0.01, epochs=epochs)
    start_time = time.perf_counter()
    model_opt.fit(X, y)
    end_time = time.perf_counter()
    opt_duration = end_time - start_time
    print(f"Optimized Fit Time: {opt_duration:10.4f}s")

    speedup = naive_duration / opt_duration if opt_duration > 0 else float('inf')
    print(f"Speedup:            {speedup:10.2f}x")

    mem = get_peak_memory()
    print(f"Peak Memory Usage:  {mem:10.2f} MB")

if __name__ == "__main__":
    print("Performance Comparison: Naive vs. Optimized Linear Regression GD")
    print("="*65)

    # Test scaling with N (Number of samples)
    print("\nScaling with Number of Samples (N):")
    for n in [1000, 10000, 100000]:
        profile_performance(n, 10, epochs=500)

    # Test scaling with D (Number of features)
    print("\nScaling with Number of Features (D):")
    for d in [10, 100, 500]:
        profile_performance(10000, d, epochs=500)
