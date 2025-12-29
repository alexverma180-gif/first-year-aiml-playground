import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "iris.csv"
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df
