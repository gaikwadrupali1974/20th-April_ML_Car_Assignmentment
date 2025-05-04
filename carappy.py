import streamlit as st
import pandas as pd
import pickle

# Load the trained model (assumes preprocessing is built-in or unnecessary)
with open("xgb_best_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚗 Car Price Prediction App")

# User Inputs
name = st.text_input("Car Name (e.g., Maruti Swift Dzire VDI)")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", value=50000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Previous Owners", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input("Mileage (kmpl)", value=18.0)
engine = st.number_input("Engine (CC)", value=1197)
max_power = st.number_input("Max Power (bhp)", value=82.0)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

# Convert inputs into DataFrame
input_dict = {
    'name': name,
    'year': year,
    'km_driven': km_driven,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"Estimated Price: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


