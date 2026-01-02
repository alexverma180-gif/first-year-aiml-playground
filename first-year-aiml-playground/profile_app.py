import cProfile
import pstats
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import load_data, train_model

def profile_app_with_caching():
    """
    Profiles the data loading and model training of the Iris app,
    running the functions multiple times to show the effect of caching.
    """
    # Initial calls to cache the results
    df = load_data()
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_model(X_train, y_train, k=5)

    # Second calls to measure performance with cache
    df_cached = load_data()
    X_cached = df_cached.drop("species", axis=1)
    y_cached = df_cached["species"]
    X_train_cached, _, y_train_cached, _ = train_test_split(X_cached, y_cached, test_size=0.2, random_state=42, stratify=y)
    train_model(X_train_cached, y_train_cached, k=5)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_app_with_caching()
    profiler.disable()

    output_file = os.path.join(os.path.dirname(__file__), 'profile_results.txt')
    with open(output_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
        stats.print_stats()

    print(f"Profiling results saved to {output_file}")
