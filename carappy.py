import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('xgb_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚗 Car Price Prediction App")

st.markdown("Fill in the car details below to predict the **selling price**.")

# Example input fields — modify these based on your dataset features 
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

# You may need to map categorical values to numbers depending on how the model was trained
fuel_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_dict = {'Dealer': 0, 'Individual': 1}
transmission_dict = {'Manual': 0, 'Automatic': 1}

# Assemble features as input array (match training format!)
input_features = np.array([[
    year, present_price, kms_driven, owner,
    fuel_dict[fuel_type], seller_dict[seller_type], transmission_dict[transmission]
]])

# Prediction
if st.button("Predict Price"):
    predicted_price = model.predict(input_features)[0]
    st.success(f"Estimated Selling Price: ₹ {predicted_price:,.2f} lakhs")
