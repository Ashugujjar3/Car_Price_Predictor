import streamlit as st
import pandas as pd
import pickle

st.title("🚗 Car Price Predictor")
st.write("Predict the selling price of a used car")

# Load model
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# User Inputs
year = st.number_input("Year of Purchase", 2000, 2026, 2018)
km_driven = st.number_input("Kilometers Driven", 0, 200000, 50000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Previous Owners", ["First Owner", "Second Owner", "Third Owner"])

if st.button("Predict Price"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Year': [year],
        'Kms_Driven': [km_driven],
        'Fuel_Type': [fuel],
        'Seller_Type': [seller],
        'Transmission': [transmission],
        'Owner': [owner]
    })
    
    input_data['Current_Year'] = 2026
    input_data['Car_Age'] = input_data['Current_Year'] - input_data['Year']
    input_data.drop('Current_Year', axis=1, inplace=True)
    
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Align columns with training data
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Selling Price: **₹{prediction:.2f} Lakhs**")
