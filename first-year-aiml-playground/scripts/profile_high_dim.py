import numpy as np
import cProfile
import pstats
import io
import sys
import os
import resource

# Ensure we can import from scratch_ml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scratch_ml.linear_regression import LinearRegressionGD

def get_peak_memory_mb():
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def profile_high_dim():
    n = 10000
    d = 2000
    epochs = 10

    print(f"Profiling high-dimensional fit (N={n}, D={d}, epochs={epochs})...")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)

    mem_initial = get_peak_memory_mb()
    print(f"Initial Peak Memory: {mem_initial:.2f} MB")

    model = LinearRegressionGD(lr=0.01, epochs=epochs)

    pr = cProfile.Profile()
    pr.enable()
    model.fit(X, y)
    pr.disable()

    mem_final = get_peak_memory_mb()
    print(f"Final Peak Memory: {mem_final:.2f} MB")
    print(f"Approximate memory increase during fit: {mem_final - mem_initial:.2f} MB")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    profile_high_dim()
