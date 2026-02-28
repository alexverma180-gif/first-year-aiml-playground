import cProfile
import pstats
import io
import sys
import os
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.model_utils import load_data, train_model

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

if __name__ == "__main__":
    profile_functions()
