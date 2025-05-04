import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize df *outside* the try-except
df = pd.DataFrame()  # Initialize as an empty DataFrame

# Load the trained XGBoost model
try:
    with open('xgb_best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please make sure 'xgb_best_model.pkl' is in the same directory.")
    st.stop()

# Load the original dataset (for unique values in dropdowns)
# Replace with your actual data loading mechanism (e.g., from CSV, database)
try:
    import pymysql
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='Reya@2014',
        port=3306
    )
    query = "SELECT * FROM cars.car_price_dataset"
    df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:  # Catch a broader range of exceptions
    st.error(f"Failed to connect to the database and load data: {e}. Using fallback data.")
    #  Minimal fallback data to allow the app to *run* (but dropdowns will be empty)
    df = pd.DataFrame({
        'Brand': ['Default'], 'Model': ['Default'], 'Year': [2020],
        'Engine_Size': [2.0], 'Fuel_Type': ['Default'], 'Transmission': ['Default'],
        'Mileage': [0], 'Doors': [4], 'Owner_Count': [1]
    })
    #  ***IMPORTANT***:  In a real application, replace this with loading from a CSV backup
    #  or some other reliable source!  Having only "Default" is not useful for prediction.


# Streamlit App
st.title('Car Price Prediction')

# Input Widgets
brand = st.selectbox('Brand', df['Brand'].unique())
model = st.selectbox('Model', df['Model'].unique())
year = st.slider('Year', int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].mean()))
engine_size = st.number_input('Engine Size', min_value=0.0, value=2.0)
fuel_type = st.selectbox('Fuel Type', df['Fuel_Type'].unique())
transmission = st.selectbox('Transmission', df['Transmission'].unique())
mileage = st.number_input('Mileage', min_value=0)
doors = st.slider('Doors', int(df['Doors'].min()), int(df['Doors'].max()), int(df['Doors'].mean()))
owner_count = st.slider('Owner Count', int(df['Owner_Count'].min()), int(df['Owner_Count'].max()),
                  int(df['Owner_Count'].mean()))

# Prepare Input Features
input_data = pd.DataFrame({
    'Brand': [brand],
    'Model': [model],
    'Year': [year],
    'Engine_Size': [engine_size],
    'Fuel_Type': [fuel_type],
    'Transmission': [transmission],
    'Mileage': [mileage],
    'Doors': [doors],
    'Owner_Count': [owner_count]
})

# Encode Categorical Features
for col in ['Brand', 'Model', 'Fuel_Type', 'Transmission']:
    le = LabelEncoder()
    le.fit(df[col])  # Important: Fit on the original data!
    input_data[col] = le.transform(input_data[col])

# Predict Price
if st.button('Predict Price'):
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f'Predicted Price: ${predicted_price:.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
