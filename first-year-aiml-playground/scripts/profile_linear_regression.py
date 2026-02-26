import time
import numpy as np
import sys
import os

# Robust path handling: Add the 'first-year-aiml-playground' directory to sys.path
# This ensures that 'scratch_ml' and 'app' can be imported regardless of the working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir if os.path.basename(script_dir) != 'scripts' else os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import resource
except ImportError:
    resource = None

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

            grad_w = (2/n) * (X.T @ error)
            grad_b = (2/n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_memory_usage():
    """Returns peak memory usage in MB."""
    if resource is None:
        return 0.0 # Memory profiling not supported on this platform
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return usage / (1024 * 1024)
    else:
        # On Linux, ru_maxrss is in kilobytes
        return usage / 1024

def profile_linear_regression():
    # 1. Performance Comparison (N=10000, D=10, 1000 epochs)
    n, d = 10000, 10
    epochs = 1000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"--- Performance Comparison (N={n}, D={d}, Epochs={epochs}) ---")

    # Naive Implementation
    start_time = time.perf_counter()
    naive_model = NaiveLinearRegressionGD(lr=0.01, epochs=epochs)
    naive_model.fit(X, y)
    naive_time = time.perf_counter() - start_time
    print(f"Naive Fit Time: {naive_time:.4f}s")

    # Optimized Implementation
    start_time = time.perf_counter()
    opt_model = LinearRegressionGD(lr=0.01, epochs=epochs)
    opt_model.fit(X, y)
    opt_time = time.perf_counter() - start_time
    print(f"Optimized Fit Time: {opt_time:.4f}s")

    speedup = naive_time / opt_time
    print(f"Speedup: {speedup:.2f}x")

    # 2. High-dimensional Memory Analysis (N=10000, D=500)
    n_high, d_high = 10000, 500
    X_high = rng.standard_normal((n_high, d_high))
    y_high = X_high @ rng.standard_normal(d_high) + rng.standard_normal(n_high)

    print(f"\n--- High-dimensional Memory Analysis (N={n_high}, D={d_high}) ---")
    # Note: ru_maxrss is the peak memory usage during the lifetime of the process.

    model_high = LinearRegressionGD(lr=0.01, epochs=100)
    model_high.fit(X_high, y_high)

    peak_mem = get_memory_usage()
    if peak_mem > 0:
        print(f"Peak Memory Usage: {peak_mem:.2f} MB")
    else:
        print("Memory profiling not available on this platform.")

    # 3. Large N Scaling (N=100000, D=10, 1000 epochs)
    n_large = 100000
    X_large = rng.standard_normal((n_large, d))
    y_large = X_large @ rng.standard_normal(d) + rng.standard_normal(n_large)

    print(f"\n--- Large N Performance (N={n_large}, D={d}, Epochs={epochs}) ---")

    start_time = time.perf_counter()
    naive_model_large = NaiveLinearRegressionGD(lr=0.01, epochs=epochs)
    naive_model_large.fit(X_large, y_large)
    naive_time_large = time.perf_counter() - start_time
    print(f"Naive Large N Fit Time: {naive_time_large:.4f}s")

    start_time = time.perf_counter()
    opt_model_large = LinearRegressionGD(lr=0.01, epochs=epochs)
    opt_model_large.fit(X_large, y_large)
    opt_time_large = time.perf_counter() - start_time
    print(f"Optimized Large N Fit Time: {opt_time_large:.4f}s")

    speedup_large = naive_time_large / opt_time_large
    print(f"Large N Speedup: {speedup_large:.2f}x")

if __name__ == "__main__":
    profile_linear_regression()
