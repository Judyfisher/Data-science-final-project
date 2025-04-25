import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os 

try:
    preprocessor_regression = joblib.load('preprocessor_regression.joblib')
    best_regression_model = joblib.load('best_lgbm_regressor.joblib')
except FileNotFoundError as e:
    st.error(f"Error: Regression preprocessor or model file not found! {e}")
    st.stop()

try:
    preprocessor_classification = joblib.load('preprocessor_classification.joblib')
    best_hybrid_classifier = joblib.load('best_hybrid_classifier.joblib')
except FileNotFoundError as e:
    st.error(f"Error: Classification preprocessor or model file not found! {e}")
    st.stop()

st.sidebar.header('Enter Manufacturing Data:')

air_temperature = st.sidebar.number_input('Air Temperature (K)', min_value=290.0, max_value=310.0, value=295.0, step=0.1)
process_temperature = st.sidebar.number_input('Process Temperature (K)', min_value=300.0, max_value=320.0, value=305.0, step=0.1)
rotational_speed = st.sidebar.number_input('Rotational Speed (rpm)', min_value=1100, max_value=2900, value=1700, step=1)
torque = st.sidebar.number_input('Torque (Nm)', min_value=3.0, max_value=100.0, value=40.0, step=0.1)
tool_wear = st.sidebar.number_input('Tool Wear (min)', min_value=0, max_value=300, value=50, step=1)
machine_type = st.sidebar.selectbox('Machine Type', ['Type_H', 'Type_L', 'Type_M'])


temp_diff = process_temperature - air_temperature
torque_squared = torque ** 2
rotational_speed_squared = rotational_speed ** 2
torque_speed_ratio = torque / (rotational_speed + 1e-9) 


torque_rolling_mean_5 = torque  
torque_rolling_mean_15 = torque 
rotational_speed_rolling_mean_5 = rotational_speed 
rotational_speed_rolling_mean_15 = rotational_speed


input_data = pd.DataFrame({
    'AirTemperature': [air_temperature],
    'ProcessTemperature': [process_temperature],
    'RotationalSpeed': [rotational_speed],
    'Torque': [torque],
    'ToolWear': [tool_wear],
    'temp_diff': [temp_diff],
    'torque_squared': [torque_squared],
    'rotational_speed_squared': [rotational_speed_squared],
    'torque_speed_ratio': [torque_speed_ratio],
    'torque_rolling_mean_5': [torque_rolling_mean_5],
    'torque_rolling_mean_15': [torque_rolling_mean_15],
    'rotational_speed_rolling_mean_5': [rotational_speed_rolling_mean_5],
    'rotational_speed_rolling_mean_15': [rotational_speed_rolling_mean_15],
    'Type_H': [1 if machine_type == 'Type_H' else 0],
    'Type_L': [1 if machine_type == 'Type_L' else 0],
    'Type_M': [1 if machine_type == 'Type_M' else 0],
    'TWF': [0], 'HDF': [0], 'PWF': [0], 'OSF': [0], 'RNF': [0]
})


if st.button('Predict'):
    
    input_data_regression = input_data[['AirTemperature', 'ProcessTemperature', 'RotationalSpeed', 'Torque', 'Type_H', 'Type_L', 'Type_M']]
    scaled_data_regression = preprocessor_regression.transform(input_data_regression)
    tool_wear_prediction = best_regression_model.predict(scaled_data_regression)[0]
    st.subheader(f'Predicted Tool Wear: {tool_wear_prediction:.2f} minutes')

    
    input_data_classification = input_data[[
        'AirTemperature', 'ProcessTemperature', 'RotationalSpeed', 'Torque', 'ToolWear',
        'temp_diff', 'torque_squared', 'rotational_speed_squared', 'torque_speed_ratio',
        'torque_rolling_mean_5',
        'torque_rolling_mean_15',
        'rotational_speed_rolling_mean_5',
        'rotational_speed_rolling_mean_15',
        'Type_L', 'Type_M', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
    ]]
    scaled_data_classification = preprocessor_classification.transform(input_data_classification)
    failure_prediction = best_hybrid_classifier.predict(scaled_data_classification)[0]

    st.subheader('Predicted Machine Failure:')
    if failure_prediction == 1:
        st.error('Machine Failure Likely!')
    else:
        st.success('No Machine Failure Predicted.')


st.markdown('---')
st.markdown('Developed by Judy')
