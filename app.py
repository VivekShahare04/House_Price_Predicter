import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the trained model and scaler
with open("linear_regression_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)


def main():
    st.title("House Price Predictor")
    st.subheader("Enter the details of the house to predict the price")

    # Input fields for user to enter house details
    area = st.number_input("Area of the house")
    bedrooms = st.number_input("Number of Bedrooms")
    bathrooms = st.number_input("Number of Bathrooms")

    # Predict button to trigger the prediction
    if st.button("Predict"):
        # Scale the input features
        scaled_features = scaler.transform([[area, bedrooms, bathrooms]])

        # Perform prediction using the loaded model
        prediction = model.predict(scaled_features)

        # Display the predicted price
        st.subheader("The predicted price of the house is")
        st.write(prediction)


if __name__ == '__main__':
    main()
