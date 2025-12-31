import streamlit as st
from sklearn.metrics import accuracy_score
from model_utils import load_data, train_model

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Classifier (Beginner AIML App)")

df = load_data()
st.write("Sample of the dataset:")
st.dataframe(df.head())

with st.sidebar:
    st.header("Model Settings")
    k = st.slider("K (neighbors)", 1, 15, 5)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)


model, X_test, y_test = train_model(df, k, test_size)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

st.subheader(f"Accuracy: {acc:.3f}")

st.write("Try a custom prediction:")
sepal_length = st.number_input("sepal_length", 4.0, 8.5, 5.1, 0.1)
sepal_width = st.number_input("sepal_width", 2.0, 4.5, 3.5, 0.1)
petal_length = st.number_input("petal_length", 1.0, 7.0, 1.4, 0.1)
petal_width = st.number_input("petal_width", 0.1, 2.5, 0.2, 0.1)

if st.button("Predict"):
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    st.success(f"Predicted species: **{pred}**")
