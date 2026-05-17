import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Iris Flower Classification")

st.write("Enter flower measurements to predict the species")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Prediction button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    species = {
        0: 'Iris-setosa',
        1: 'Iris-versicolor',
        2: 'Iris-virginica'
    }

    st.success(f"Predicted Species: {species[prediction[0]]}")
