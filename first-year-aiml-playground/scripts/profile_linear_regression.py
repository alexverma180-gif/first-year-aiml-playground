import numpy as np
import time
import sys
import os
try:
    import resource
except ImportError:
    resource = None

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

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
            # Naive O(N*D) gradient calculation
            y_pred = X @ self.w + self.b
            error = y_pred - y

            grad_w = (2/n) * (X.T @ error)
            grad_b = (2/n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_peak_memory():
    # Returns peak memory in MB
    if resource:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0.0

def benchmark_models(n, d, epochs=100):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"\nBenchmarking with N={n}, D={d}, Epochs={epochs}")

    # Naive Model
    start_mem = get_peak_memory()
    start_time = time.perf_counter()
    NaiveLinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    naive_time = time.perf_counter() - start_time
    naive_mem = get_peak_memory() - start_mem
    print(f"Naive GD: Time = {naive_time:.4f}s, Peak Mem Delta = {naive_mem:.4f} MB")

    # Optimized Model
    start_mem = get_peak_memory()
    start_time = time.perf_counter()
    LinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    opt_time = time.perf_counter() - start_time
    opt_mem = get_peak_memory() - start_mem
    print(f"Optimized GD: Time = {opt_time:.4f}s, Peak Mem Delta = {opt_mem:.4f} MB")

    print(f"Speedup: {naive_time / opt_time:.2f}x")

if __name__ == "__main__":
    print("Linear Regression GD Performance & Memory Analysis")
    benchmark_models(10000, 10, 1000)
    benchmark_models(100000, 10, 1000)
    benchmark_models(10000, 500, 1000)
