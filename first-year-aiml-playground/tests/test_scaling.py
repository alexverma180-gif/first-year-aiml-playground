import pytest
import numpy as np
from scratch_ml.linear_regression import LinearRegressionGD

def generate_data(n, d, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-10, 10, size=(n, d))
    true_w = rng.uniform(-1, 1, size=d)
    true_b = 5.0
    y = X @ true_w + true_b + rng.normal(0, 0.5, size=n)
    return X, y

@pytest.mark.parametrize("n", [1000, 10000])
@pytest.mark.parametrize("d", [10, 50])
def test_linear_regression_scaling(benchmark, n, d):
    """Benchmark LinearRegressionGD.fit with varying N and D."""
    X, y = generate_data(n, d)
    model = LinearRegressionGD(lr=0.01, epochs=1000) # Increased epochs to emphasize loop performance
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)
