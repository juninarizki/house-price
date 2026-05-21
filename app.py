import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ====== PERBAIKAN DI SINI ======
# Mengubah st.set_page_title menjadi st.set_page_config
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="centered"
)
# ===============================

# Load the saved model and scaler
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App header
st.title("🏠 House Price Predictor")
st.write("Enter the details of the house below to get an estimated price.")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    sq_ft = st.number_input("Square Footage", min_value=100, max_value=10000, value=2500)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2005)

with col2:
    lot_size = st.number_input("Lot Size (Acres)", min_value=0.1, max_value=20.0, value=1.5)
    garage_size = st.selectbox("Garage Size (Number of Cars)", [0, 1, 2])
    quality = st.slider("Neighborhood Quality (1-10)", 1, 10, 8)

# Prediction button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame([[sq_ft, bedrooms, bathrooms, year_built, lot_size, garage_size, quality]], 
                              columns=['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Display result
    st.success(f"The estimated price for this house is: ${prediction[0]:,.2f}")
