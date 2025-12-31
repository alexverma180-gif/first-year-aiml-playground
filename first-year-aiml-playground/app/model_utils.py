import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "iris.csv"
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource
def train_model(df, k, test_size):
    """Splits data and trains a KNeighborsClassifier, caching the model."""
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model, X_test, y_test
