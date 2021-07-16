from math import isnan
from pandas.core.frame import DataFrame
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures


def load_model():
    with open('car_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_data():
    with open('car_d.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_x():
    with open('car_x.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_y():
    with open('car_y.pkl', 'rb') as file:
        data = pickle.load(file)
    return data





data = load_model()
dataf = load_data()
datax = load_x()
datay = load_y()
df = pd.DataFrame(dataf['df'])
regressor_model = data['model']
regressor_Xtrain = datax['X_train']
regressor_ytrain = datay['y_train']

def show_predict_page():
    st.title("Car Predictions")
    st.write('''### Enter Vehicle information here''')
    year = st.number_input(label='year' , step=1)
    make = st.selectbox('make',sorted(df[df['year']== year]['make_name'].value_counts().index.tolist()))
    model = st.selectbox('model', sorted (df[(df['make_name'] == make) & (df['year'] == year)]['model_name'].value_counts().index.tolist()))
    trim = st.selectbox('trim', sorted(df[(df['model_name'] == model) & (df['make_name'] == make) & (df['year'] == year)]['trim_name'].value_counts().index.tolist()))
    mileage = st.number_input(label='mileage' , step=1)
    horsepower = df[(df['trim_name']==trim)].horsepower.mean()
    engine_displacement = df[(df['trim_name']==trim)].engine_displacement.mean()
    fuel_tank_volume = df[df['trim_name']==trim].fuel_tank_volume.mode()
    wheelbase = df[df['trim_name']==trim].wheelbase.mean()

    calculate = st.button('Calculate')

    def check_for_input():
        if (make == None or model == None or trim == None or mileage == None or year == None):
            return False
        else:
            return True

    def run_model():
        to_scale = ['mileage', 'horsepower', 'engine_displacement','fuel_tank_volume','wheelbase']
        scal = [[mileage, horsepower, engine_displacement,fuel_tank_volume, wheelbase]]
        robust_scaler = RobustScaler()
        robust_scaler.fit(df[to_scale])
        sc = robust_scaler.transform(scal)


        mileage_s = sc[0][0]
        horsepower_s = sc[0][1]
        engine_displacement_s = sc[0][2]
        fuel_tank_volume_s =sc[0][3]
        wheelbase_s = sc[0][4]


        make_num = df[df['make_name'] == make]['make_name_num'].mean()
        model_num = df[df['model_name'] == model]['model_name_num'].mean()
        trim_num = df[df['trim_name'] == trim]['trimId'].mean()

        X = [[make_num, model_num, trim_num, year, mileage_s, horsepower_s,fuel_tank_volume_s,engine_displacement_s,wheelbase_s]]

        pf = PolynomialFeatures(degree=3)
         
        X_train_degree = pf.fit_transform(regressor_Xtrain)
        X_degree = pf.transform(X)

        regressor_model.fit(X_train_degree ,regressor_ytrain)
        result = regressor_model.predict(X_degree)

        return round(result[0],2)

    if calculate:
        if check_for_input() == False:
            st.write('''### Please enter all the above information''')
        else:
           estimated_price = run_model()
           st.write(''' ### ${}'''.format(estimated_price))
