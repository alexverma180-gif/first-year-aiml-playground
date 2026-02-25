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
            y_pred = X @ self.w + self.b
            # Standard gradient descent updates
            dw = (2.0/n) * (X.T @ (y_pred - y))
            db = (2.0/n) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

def get_peak_memory():
    # Returns peak memory usage in MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def profile_linear_regression():
    print("=== LinearRegressionGD Performance Analysis ===")

    # Configuration for standard benchmark
    N = 10000
    D = 10
    epochs = 1000
    lr = 0.01

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D))
    y = X @ rng.standard_normal(D) + rng.standard_normal(N)

    print(f"Dataset: N={N}, D={D}, Epochs={epochs}")

    # Profile Naive Implementation
    naive_model = NaiveLinearRegressionGD(lr=lr, epochs=epochs)
    start_time = time.perf_counter()
    naive_model.fit(X, y)
    naive_duration = time.perf_counter() - start_time
    print(f"Naive Fit Duration: {naive_duration:.4f} seconds")

    # Profile Optimized Implementation
    optimized_model = LinearRegressionGD(lr=lr, epochs=epochs)
    start_time = time.perf_counter()
    optimized_model.fit(X, y)
    optimized_duration = time.perf_counter() - start_time
    print(f"Optimized Fit Duration: {optimized_duration:.4f} seconds")

    speedup = naive_duration / optimized_duration
    print(f"Speedup: {speedup:.2f}x")

    # High-dimensional Memory and Performance Analysis
    print("\n--- High-Dimensional Scaling (N=10000, D=500, Epochs=100) ---")
    D_high = 500
    X_high = rng.standard_normal((N, D_high))
    y_high = X_high @ rng.standard_normal(D_high) + rng.standard_normal(N)

    # Baseline memory
    _ = get_peak_memory()

    start_time = time.perf_counter()
    optimized_model_high = LinearRegressionGD(lr=lr, epochs=100)
    optimized_model_high.fit(X_high, y_high)
    high_d_duration = time.perf_counter() - start_time
    peak_mem = get_peak_memory()

    print(f"High-D Fit Duration: {high_d_duration:.4f} seconds")
    print(f"Peak Memory Usage: {peak_mem:.2f} MB")

    # Scaling with N
    print("\n--- Scaling with N (D=10, Epochs=100) ---")
    for n_size in [1000, 10000, 100000]:
        X_n = rng.standard_normal((n_size, 10))
        y_n = X_n @ rng.standard_normal(10) + rng.standard_normal(n_size)
        start_time = time.perf_counter()
        LinearRegressionGD(lr=lr, epochs=100).fit(X_n, y_n)
        duration = time.perf_counter() - start_time
        print(f"N={n_size:6d}: {duration*1000:.2f} ms")

if __name__ == "__main__":
    profile_linear_regression()
