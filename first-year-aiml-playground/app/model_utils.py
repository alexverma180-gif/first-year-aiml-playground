import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "iris.csv"
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_model(X_train, y_train, k):
    """Trains a KNeighborsClassifier model."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model
