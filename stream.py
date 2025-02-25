import numpy as np
import joblib
import streamlit as st


#load the trained model
model = joblib.load('model_scaled.pkl')
scale = joblib.load('scaler_scaled.pkl')

#streamlit app title
st.title('Machine Learning Model Deployment')
st.write('Enter your Medical Details to know about your Diabetic status')

#define the input fields
st.sidebar.header('Your Medical Records')

preg = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
plas = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
pres = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
skin = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
test = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
mass = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
pedi = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)
age  = st.sidebar.number_input('preg', min_value = 0.0, max_value= 100.0, value=50.0, step=0.1)

input_data = np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
scaled_input = scale.transform(input_data)

if st.sidebar.button('Predict'):
    prediction = model.predict(scaled_input)
    st.success(f'Prediction: {prediction[0]}')

if prediction == 0:
    st.success('You are Diabetic')
else:
    st.success('You are not Diabetic')



