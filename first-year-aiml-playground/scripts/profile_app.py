import cProfile
import pstats
import io
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.path.abspath("first-year-aiml-playground"))

from app.model_utils import load_data, train_model
from scratch_ml.linear_regression import LinearRegressionGD
import numpy as np

def profile_functions():
    # Profiling load_data
    print("Profiling load_data...")
    pr = cProfile.Profile()
    pr.enable()
    df = load_data.__wrapped__()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())

    # Profiling train_model
    print("Profiling train_model...")
    X = df.drop("species", axis=1)
    y = df["species"]

    pr = cProfile.Profile()
    pr.enable()
    model = train_model.__wrapped__(X, y, k=5)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())

    # Profiling LinearRegressionGD.fit
    print("Profiling LinearRegressionGD.fit...")
    rng = np.random.default_rng(42)
    num_samples = 10000
    X_lr = rng.uniform(-10, 10, size=(num_samples, 3))
    y_lr = 2 * X_lr[:, 0] + 3 * X_lr[:, 1] - 5 * X_lr[:, 2] + 7 + rng.normal(0, 0.5, size=num_samples)

    model_lr = LinearRegressionGD(lr=0.01, epochs=1000)

    pr = cProfile.Profile()
    pr.enable()
    model_lr.fit(X_lr, y_lr)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())

if __name__ == "__main__":
    profile_functions()
