import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('xgb_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
owner_count = st.slider('Owner Count', int(df['Owner_Count'].min()), int(df['Owner_Count'].max()), int(df['Owner_Count'].mean()))


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
