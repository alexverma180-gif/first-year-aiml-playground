import numpy as np
import time
import sys
import os

# Ensure we can import from scratch_ml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
            # Naive O(N*D) calculation per iteration
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (2.0 / n) * (X.T @ error)
            db = (2.0 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

def compare():
    ns = [1000, 10000, 100000]
    d = 10
    epochs = 100
    lr = 0.01

    print(f"{'N':>10} | {'Naive (ms)':>12} | {'Optimized (ms)':>14} | {'Speedup':>10}")
    print("-" * 55)

    rng = np.random.default_rng(42)

    for n in ns:
        X = rng.standard_normal((n, d))
        y = X @ rng.standard_normal(d) + rng.standard_normal(n)

        # Naive
        naive_model = NaiveLinearRegressionGD(lr=lr, epochs=epochs)
        # Warmup
        naive_model.fit(X, y)

        start = time.perf_counter()
        for _ in range(5):
            naive_model.fit(X, y)
        naive_time = (time.perf_counter() - start) / 5 * 1000 # ms

        # Optimized
        opt_model = LinearRegressionGD(lr=lr, epochs=epochs)
        # Warmup
        opt_model.fit(X, y)

        start = time.perf_counter()
        for _ in range(5):
            opt_model.fit(X, y)
        opt_time = (time.perf_counter() - start) / 5 * 1000 # ms

        speedup = naive_time / opt_time
        print(f"{n:10,d} | {naive_time:12.2f} | {opt_time:14.2f} | {speedup:9.2f}x")

if __name__ == "__main__":
    compare()
