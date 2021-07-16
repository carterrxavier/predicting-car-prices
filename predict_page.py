<<<<<<< HEAD
from math import isnan
=======
>>>>>>> 1ded3024cfced68134c28c87f2ba9cce0bd9533a
import streamlit as st
import pickle
import numpy as np
import pandas as pd
<<<<<<< HEAD
from sklearn.preprocessing import RobustScaler
=======
>>>>>>> 1ded3024cfced68134c28c87f2ba9cce0bd9533a

def load_model():
    with open('car_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
df = pd.DataFrame(data['df'])
<<<<<<< HEAD
edf = pd.DataFrame(data['edf'])
tdf = pd.DataFrame(data['tdf'])
=======
>>>>>>> 1ded3024cfced68134c28c87f2ba9cce0bd9533a
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

<<<<<<< HEAD


=======
>>>>>>> 1ded3024cfced68134c28c87f2ba9cce0bd9533a
def show_predict_page():
    st.title("Car Predictions")
    st.write('''### Enter Vehicle information here''')
    year = st.number_input(label='year' , step=1)
    make = st.selectbox('make',sorted(df[df['year']== year]['make_name'].value_counts().index.tolist()))
    model = st.selectbox('model', sorted (df[(df['make_name'] == make) & (df['year'] == year)]['model_name'].value_counts().index.tolist()))
    trim = st.selectbox('trim', sorted(df[(df['model_name'] == model) & (df['make_name'] == make) & (df['year'] == year)]['trim_name'].value_counts().index.tolist()))
<<<<<<< HEAD
    mileage = st.number_input(label='mileage' , step=1)
    horsepower = df[(df['model_name'] == model) & (df['make_name'] == make) & (df['year'] == year)].horsepower.mean()
    engine_displacement = df[(df['model_name'] == model) & (df['make_name'] == make) & (df['year'] == year)].engine_displacement.mean()

    calculate = st.button('Calculate')

    def check_for_input():
        if (make == None or model == None or trim == None or mileage == None or year == None):
            return False
        else:
            return True

    def run_model():
        to_scale = ['mileage', 'horsepower', 'engine_displacement']
        scal = [[mileage, horsepower, engine_displacement]]
        robust_scaler = RobustScaler()
        robust_scaler.fit(edf[to_scale])
        sc = robust_scaler.transform(scal)

        mileage_s = sc[0][0]
        horsepower_s = sc[0][1]
        engine_displacement_s = sc[0][2]


        st.write(edf[edf['make_name'] == make]['make_name_num'].head(1).make_name_)
        model_num = 0


        regressor_model.fit(regressor_Xtrain, regressor_ytrain)
        result = regressor_model.predict()

        return 0

    if calculate:
        if check_for_input() == False:
            st.write('''### Please enter all the above information''')
        else:
           estimated_price = run_model()


         

           
  


        

    

=======
>>>>>>> 1ded3024cfced68134c28c87f2ba9cce0bd9533a
    

