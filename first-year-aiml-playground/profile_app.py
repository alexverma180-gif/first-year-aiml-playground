import cProfile
import pstats
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import load_data, train_model

def run_data_and_model_funcs():
    """Helper function to run the core logic."""
    df = load_data()
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_model(X_train, y_train, k=5)

if __name__ == "__main__":
    # --- Profile the initial (uncached) run ---
    print("Profiling UNCACHED function calls...")
    profiler_uncached = cProfile.Profile()
    profiler_uncached.enable()

    run_data_and_model_funcs()

    profiler_uncached.disable()

    output_file_uncached = os.path.join(os.path.dirname(__file__), 'profile_results_uncached.txt')
    pstats.Stats(profiler_uncached).dump_stats('profile_results_uncached.pstats')
    with open(output_file_uncached, 'w') as f:
        stats_uncached = pstats.Stats(profiler_uncached, stream=f).sort_stats("cumtime")
        stats_uncached.print_stats()
    print(f"Uncached profiling results saved to {output_file_uncached}")

    print("-" * 50)

    # --- Profile the second (cached) run ---
    print("Profiling CACHED function calls...")
    profiler_cached = cProfile.Profile()
    profiler_cached.enable()

    run_data_and_model_funcs()

    profiler_cached.disable()

    output_file_cached = os.path.join(os.path.dirname(__file__), 'profile_results_cached.txt')
    pstats.Stats(profiler_cached).dump_stats('profile_results_cached.pstats')
    with open(output_file_cached, 'w') as f:
        stats_cached = pstats.Stats(profiler_cached, stream=f).sort_stats("cumtime")
        stats_cached.print_stats()
    print(f"Cached profiling results saved to {output_file_cached}")
