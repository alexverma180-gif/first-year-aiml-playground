import pytest
import numpy as np
from scratch_ml.linear_regression import LinearRegressionGD

@pytest.mark.benchmark(group="scaling_n")
@pytest.mark.parametrize("n", [1000, 10000, 100000])
def test_fit_scaling_n(benchmark, n):
    """Benchmark fit performance as number of samples (N) increases."""
    d = 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    # Using small number of epochs to focus on per-iteration and precomputation cost
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)

@pytest.mark.benchmark(group="scaling_d")
@pytest.mark.parametrize("d", [5, 50, 500])
def test_fit_scaling_d(benchmark, d):
    """Benchmark fit performance as number of features (D) increases."""
    n = 10000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    model = LinearRegressionGD(lr=0.01, epochs=100)
    benchmark.pedantic(model.fit, args=(X, y), rounds=5)
