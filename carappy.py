import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('xgb_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚗 Car Price Prediction App")

st.markdown("Fill in the car details below to predict the **selling price**.")

# Example input fields — modify these based on your dataset features
brand = st.selectbox("Brand", ['Toyota', 'Honda', 'Ford','Kia','Audi'])
model = st.text_input("Model")
year = st.number_input("Year", min_value=1990, max_value=2025)
mileage = st.number_input("Mileage")
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
engine_size = st.number_input("Engine Size (L)")
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
color = st.selectbox("Color", ['Red', 'Blue', 'Black', 'White'])

input_df = pd.DataFrame([[brand, model, year, mileage, fuel_type, engine_size, transmission, color, location]],
                        columns=['brand', 'model', 'year', 'mileage', 'fuel_type', 'engine_size', 'transmission', 'color', 'location'])


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
prediction = model.predict(input_df)
st.write(f"Predicted car price: ₹{prediction[0]:,.2f}")
