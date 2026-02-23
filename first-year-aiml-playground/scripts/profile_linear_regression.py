import numpy as np
import time
import resource
import sys
import os

# Add the project root to sys.path to allow for correct module imports
sys.path.append(os.path.abspath("first-year-aiml-playground"))

from scratch_ml.linear_regression import LinearRegressionGD

class NaiveLinearRegressionGD:
    """
    A naive implementation of Gradient Descent for Linear Regression.
    Recalculates the gradient using the entire dataset in each iteration.
    Complexity per iteration: O(N*D).
    """
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
            # Naive O(N*D) gradient calculation: Y_pred = Xw + b
            y_pred = X @ self.w + self.b
            error = y_pred - y

            # Gradient of w: (2/n) * X.T @ error
            grad_w = (2.0 / n) * (X.T @ error)
            # Gradient of b: (2/n) * sum(error)
            grad_b = (2.0 / n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

def get_memory_usage():
    """Returns the peak resident set size in MB."""
    # resource.getrusage returns values in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def profile_linear_regression():
    # Parameters for baseline comparison
    n, d = 10000, 3
    epochs = 1000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    print(f"--- Profiling Linear Regression GD (N={n}, D={d}, Epochs={epochs}) ---")

    # Measure Optimized implementation
    mem_before_opt = get_memory_usage()
    start_time_opt = time.perf_counter()
    model_opt = LinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    end_time_opt = time.perf_counter()
    mem_after_opt = get_memory_usage()
    opt_time_ms = (end_time_opt - start_time_opt) * 1000

    print(f"Optimized Fit: {opt_time_ms:.2f} ms")
    print(f"Current Peak Memory: {mem_after_opt:.2f} MB")

    # Measure Naive implementation
    mem_before_naive = get_memory_usage()
    start_time_naive = time.perf_counter()
    model_naive = NaiveLinearRegressionGD(lr=0.01, epochs=epochs).fit(X, y)
    end_time_naive = time.perf_counter()
    mem_after_naive = get_memory_usage()
    naive_time_ms = (end_time_naive - start_time_naive) * 1000

    print(f"Naive Fit:     {naive_time_ms:.2f} ms")
    print(f"Current Peak Memory: {mem_after_naive:.2f} MB")

    if opt_time_ms > 0:
        print(f"Speedup: {naive_time_ms / opt_time_ms:.1f}x")

    # High-dimensional performance and memory analysis
    print("\n--- High-Dimensional Analysis (N=10000, D=500, Epochs=100) ---")
    n_high, d_high = 10000, 500
    X_high = rng.standard_normal((n_high, d_high))
    y_high = X_high @ rng.standard_normal(d_high) + rng.standard_normal(n_high)

    start_time_high = time.perf_counter()
    LinearRegressionGD(lr=0.01, epochs=100).fit(X_high, y_high)
    end_time_high = time.perf_counter()

    print(f"Optimized Fit (D=500): {(end_time_high - start_time_high) * 1000:.2f} ms")
    print(f"Final Peak Memory Usage: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    profile_linear_regression()
