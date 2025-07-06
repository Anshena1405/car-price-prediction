import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model pipeline
with open("LinearRegressionModel.pkl", "rb") as f:
    pipe = pickle.load(f)

# Title
st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Car Price Prediction App")

# User Inputs
name = st.selectbox("Select Car Model", pipe.named_steps['columntransformer'].transformers_[0][1].categories_[0])
company = st.selectbox("Select Company", pipe.named_steps['columntransformer'].transformers_[0][1].categories_[1])
fuel_type = st.selectbox("Select Fuel Type", pipe.named_steps['columntransformer'].transformers_[0][1].categories_[2])
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2018)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=25000)

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=[[name, company, year, kms_driven, fuel_type]])
    
    price = int(pipe.predict(input_df)[0])
    st.success(f"Estimated Car Price: ₹ {price:,}")
