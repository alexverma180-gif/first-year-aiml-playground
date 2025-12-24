import time
import numpy as np
import sys
import os

# Add the parent directory to the path to import the LinearRegressionGD class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch_ml.linear_regression import LinearRegressionGD

def run_performance_test():
    """
    Tests the performance of the LinearRegressionGD model.
    """
    print("Starting performance test...")

    # Generate a large dataset
    rng = np.random.default_rng(42)
    num_samples = 10000
    X = rng.uniform(-10, 10, size=(num_samples, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + 7 + rng.normal(0, 0.5, size=num_samples)

    print(f"Generated a dataset with {num_samples} samples.")

    # Initialize the model
    model = LinearRegressionGD(lr=0.01, epochs=1000)

    # Time the fit method
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    fit_time = end_time - start_time
    print(f"Execution time for fit method: {fit_time:.4f} seconds")

    # Time the predict method
    start_time = time.time()
    model.predict(X)
    end_time = time.time()
    predict_time = end_time - start_time
    print(f"Execution time for predict method: {predict_time:.4f} seconds")

    print("Performance test finished.")

if __name__ == "__main__":
    run_performance_test()
