import cProfile
import pstats
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import load_data, train_model

def profile_app():
    """Profiles the data loading and model training of the Iris app."""
    df = load_data()
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_model(X_train, y_train, k=5)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_app()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)
