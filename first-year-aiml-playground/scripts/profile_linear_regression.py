import cProfile
import pstats
import io
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath("first-year-aiml-playground"))

from scratch_ml.linear_regression import LinearRegressionGD

def profile_fit():
    print("Profiling LinearRegressionGD.fit...")
    n = 10000
    d = 20
    epochs = 10000 # Increase epochs to make bottlenecks more apparent
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)

    model = LinearRegressionGD(epochs=epochs)

    pr = cProfile.Profile()
    pr.enable()
    model.fit(X, y)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    profile_fit()
