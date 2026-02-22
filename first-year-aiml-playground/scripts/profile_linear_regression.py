import time
import numpy as np
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
            # Standard O(N*D) gradient calculation
            y_pred = X @ self.w + self.b
            error = y_pred - y

            # Use NumPy vectorization for standard GD
            grad_w = (2/n) * (X.T @ error)
            grad_b = (2/n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_peak_memory():
    # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def run_profile(n, d, epochs=1000):
    print(f"\n--- Profiling: N={n}, D={d}, Epochs={epochs} ---")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    # Naive Implementation
    mem_before = get_peak_memory()
    start_time = time.perf_counter()
    naive_model = NaiveLinearRegressionGD(lr=0.01, epochs=epochs)
    naive_model.fit(X, y)
    end_time = time.perf_counter()
    mem_after = get_peak_memory()
    naive_time = end_time - start_time
    naive_mem = mem_after - mem_before
    print(f"Naive GD:     Time = {naive_time:.4f}s, Peak RSS Delta = {naive_mem} KB")

    # Optimized Implementation
    mem_before = get_peak_memory()
    start_time = time.perf_counter()
    opt_model = LinearRegressionGD(lr=0.01, epochs=epochs)
    opt_model.fit(X, y)
    end_time = time.perf_counter()
    mem_after = get_peak_memory()
    opt_time = end_time - start_time
    opt_mem = mem_after - mem_before
    print(f"Optimized GD: Time = {opt_time:.4f}s, Peak RSS Delta = {opt_mem} KB")

    speedup = naive_time / opt_time if opt_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Test varying N (D constant)
    print("=== Testing Scaling with N (D=10) ===")
    for n in [1000, 10000, 100000]:
        run_profile(n, 10, epochs=1000)

    # Test varying D (N constant)
    print("\n=== Testing Scaling with D (N=10000) ===")
    for d in [10, 100, 500]:
        run_profile(10000, d, epochs=1000)
