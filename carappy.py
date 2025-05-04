st.title('Car Price Prediction')

# **CRITICAL: Data Type Handling for 'Year'**
if not df.empty:
    try:
        df['Year'] = pd.to_numeric(df['Year'], errors='raise')  # Convert to numeric, raise errors
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        mean_year = int(df['Year'].mean())
    except ValueError:
        st.error("The 'Year' column contains non-numeric values and cannot be used for the slider.")
        st.stop()  # Stop execution if 'Year' is invalid
    except TypeError:
        st.error("The 'Year' column contains incompatible data types for calculations.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the 'Year' column: {e}")
        st.stop()

else:
    # Handle the case where df is empty (from fallback or database failure)
    min_year = 2000
    max_year = 2025
    mean_year = 2020
    st.warning("No data loaded. Using default year range for the slider.")


# Input Widgets
brand = st.selectbox('Brand', df['Brand'].unique() if not df.empty else ['Default'])
model = st.selectbox('Model', df['Model'].unique() if not df.empty else ['Default'])
year = st.slider('Year', min_year, max_year, mean_year)  # Use calculated or default values
engine_size = st.number_input('Engine Size', min_value=0.0, value=2.0)
fuel_type = st.selectbox('Fuel Type', df['Fuel_Type'].unique() if not df.empty else ['Default'])
transmission = st.selectbox('Transmission', df['Transmission'].unique() if not df.empty else ['Default'])
mileage = st.number_input('Mileage', min_value=0)
doors = st.slider('Doors', 2, 5, 4)  # Provide default values
owner_count = st.slider('Owner Count', 1, 5, 2)  # Provide default values


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
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
