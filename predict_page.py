import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('car_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
df = pd.DataFrame(data['df'])
regressor_model = data['model']
regressor_Xtrain = data['poly']
regressor_ytrain = data['ytrain']
year = data['year']
car_make = data['car_make']
car_model = data['car_model']
mileage = data['mileage']
trimId = data['trimId']
horsepower = data['horsepower']
engine_displacement = data['engine_displacement']

def show_predict_page():
    st.title("Car Predictions")
    st.write('''### Enter Vehicle information here''')
    year = st.number_input(label='year' , step=1)
    make = st.selectbox('make',sorted(df[df['year']== year]['make_name'].value_counts().index.tolist()))
    model = st.selectbox('model', sorted (df[(df['make_name'] == make) & (df['year'] == year)]['model_name'].value_counts().index.tolist()))
    trim = st.selectbox('trim', sorted(df[(df['model_name'] == model) & (df['make_name'] == make) & (df['year'] == year)]['trim_name'].value_counts().index.tolist()))
    

