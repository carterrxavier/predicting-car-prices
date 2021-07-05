import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_for_model(df):
    '''
    This function take in the telco_churn data acquired,
    performs a split into 3 dataframes. one for train, one for validating and one for testing 
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=765)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=231)
    
    print('train{},validate{},test{}'.format(train.shape, validate.shape, test.shape))
    return train, validate, test



def summerize_df(df):
    print('-----shape------')
    print('{} rows and {} columns'.format(df.shape[0], df.shape[1]))
    print('---info---')
    print(df.info())
    print(df.describe())
    print('----Catagorical Variables----')
    print(df.select_dtypes(include='object').columns.tolist())
    print('----Continous  Variables----')
    print(df.select_dtypes(exclude='object').columns.tolist())
    
    print('--nulls--')
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    print(df.isna().sum())
    
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    
#remove nulls and columns based on %
##############################################################
##############################################################
##############################################################
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def get_nulls(df):
    col =  nulls_by_col(df)
    row =  nulls_by_row(df)
    
    return col, row


def drop_null_columns(df , null_min , col_missing = 'percent_rows_missing'):
    cols = get_nulls(df)[0]
    for i in range(len(cols)):
        if cols[col_missing][i] >= null_min:
            df = df.drop(columns = cols.index[i])
        
    return df

def drop_null_rows(df , percentage):
    min_count = int(((100-percentage)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh = min_count)
        
    return df

def drop_nulls(df, axis, percentage):
    if axis == 0:
        df = drop_null_rows(df, percentage)   
    else:
        df = drop_null_columns(df, percentage)
    return df


def get_car_data(sample= None):
    if os.path.isfile('cars_sample.csv'):
        cars_df =   pd.read_csv('cars_sample.csv')
        return cars_df
    else:
        all_cars_df = pd.read_csv('used_cars_data.csv')
        if(sample==None):
            return all_cars_df
        else:
            cars_df = all_cars_df.sample(sample ,random_state=145)
            cars_df.to_csv('cars_sample.csv', index=False)
            return cars_df
        
        
def clean_car_data(cars_df):
    cars_df = drop_null_columns(cars_df , 90)
    cars_df = drop_null_rows(cars_df, 40)
    cars_df = cars_df.drop(columns=['bed_height', 'bed_length'])
    cars_df = cars_df.dropna(axis=0, subset=['body_type'])
    split = cars_df['back_legroom'].str.split(' ', n =1, expand = True)
    split1 = cars_df['front_legroom'].str.split(' ', n =1, expand = True)
    cars_df['back_legroom'] = split[0]
    cars_df['front_legroom'] = split1[0]
    cars_df['back_legroom'] = pd.to_numeric(cars_df['back_legroom'], errors='coerce')       
    cars_df['front_legroom'] = pd.to_numeric(cars_df['front_legroom'], errors='coerce')
    cars_df['back_legroom'] = cars_df.groupby('body_type').back_legroom.transform(lambda x  : x.fillna(round(x.mean(),1)))
    cars_df['front_legroom'] = cars_df.groupby('body_type').front_legroom.transform(lambda x : x.fillna(round(x.mean(),1)))
    cars_df['city_fuel_economy'] = cars_df.groupby(['year','make_name','model_name']).city_fuel_economy.transform(lambda x : x.fillna(round(x.mean(),1)))
    cars_df = cars_df.dropna(axis=0, subset=['city_fuel_economy'])
    cars_df['highway_fuel_economy']= cars_df.groupby(['year','make_name','model_name']).highway_fuel_economy.transform(lambda x : x.fillna(round(x.mean(),1)))
    cars_df = cars_df.dropna(axis=0, subset=['highway_fuel_economy'])
    cars_df = cars_df.drop(columns='description')
    cars_df['engine_cylinders'] = cars_df.groupby(['year','make_name','model_name']).engine_cylinders.\
transform(lambda x : x.fillna(x.mode()))
    cars_df = cars_df.dropna(axis=0, subset=['engine_cylinders'])
    cars_df['engine_displacement'] = cars_df.groupby(['year','make_name','model_name']).engine_displacement.\
transform(lambda x : x.fillna(x.mode()))
    cars_df = cars_df.dropna(axis=0, subset=['engine_displacement'])
    cars_df['engine_type'] = cars_df.groupby(['year','make_name','model_name']).engine_type.\
transform(lambda x : x.fillna(x.mode()))
    cars_df = cars_df.dropna(axis=0, subset=['engine_type'])
    cars_df.fleet = cars_df.fleet.fillna('unknown')
    cars_df.frame_damaged = cars_df.frame_damaged.fillna(False)
    cars_df = cars_df.drop(columns='franchise_make')
    cars_df['fuel_type'] = cars_df.groupby(['year','make_name','model_name']).fuel_type.transform(lambda x : x.fillna(x.mode()))
    cars_df = cars_df.dropna(axis=0, subset=['fuel_type'])
    split2 = cars_df['fuel_tank_volume'].str.split(' ', n =1, expand = True)
    cars_df['fuel_tank_volume'] = split2[0]
    cars_df['fuel_tank_volume'] = pd.to_numeric(cars_df['fuel_tank_volume'], errors='coerce')
    cars_df = cars_df.dropna(axis=0, subset=['fuel_tank_volume'])
    cars_df['fuel_tank_volume'] = cars_df.groupby(['fuel_tank_volume']).fuel_tank_volume.transform(lambda x : x.fillna(round(x.mean(),1)))
    cars_df.has_accidents = cars_df.has_accidents.fillna(False)
    split3 = cars_df['height'].str.split(' ', n =1, expand = True)
    cars_df['height'] = split3[0]
    cars_df['height'] = pd.to_numeric(cars_df['height'], errors='coerce')
    cars_df = cars_df.dropna(axis=0, subset=['height'])
    cars_df.height = cars_df.groupby(['year','make_name','model_name']).height.\
    transform(lambda x : x.fillna(x.mean()))
    cars_df = cars_df.drop(columns=['isCab','main_picture_url','major_options'])
    cars_df = cars_df.dropna(axis=0, subset=['mileage'])
    cars_df = cars_df.dropna(axis=0, subset=['owner_count'])
    cars_df = cars_df.drop(columns=['power','torque'])
    cars_df = cars_df.drop(columns=['salvage','seller_rating','theft_title'])
    cars_df = cars_df.drop(columns=['sp_name'])
    cars_df = cars_df.dropna(axis=0, subset=['transmission'])
    cars_df = cars_df.dropna(axis=0, subset=['transmission_display'])
    #anything left over will be removed
    cars_df = cars_df.dropna()
    #### removing redundant columns, datatype conversions#########
    #Length, Width, Wheelbase need to be split from unit of measure to become floats
    split = cars_df['width'].str.split(' ', n =1, expand = True)
    split2 = cars_df['wheelbase'].str.split(' ', n =1, expand = True)
    split3 = cars_df['length'].str.split(' ', n =1, expand = True)
    cars_df['width'] = split[0]
    cars_df['wheelbase'] = split2[0]
    cars_df['length'] = split3[0]
    cars_df['width'] = pd.to_numeric(cars_df['width'], errors='coerce')
    cars_df['wheelbase'] = pd.to_numeric(cars_df['wheelbase'], errors='coerce')
    cars_df['length'] = pd.to_numeric(cars_df['length'], errors='coerce')
    cars_df.listed_date = pd.to_datetime(cars_df.listed_date)
    cars_df.daysonmarket = (cars_df.listed_date.max() - cars_df.listed_date).astype(str)
    split4 = cars_df['daysonmarket'].str.split(' ', n =1, expand = True)
    cars_df['daysonmarket'] = split4[0]
    cars_df['daysonmarket'] = pd.to_numeric(cars_df['daysonmarket'], errors='coerce')
    cars_df = cars_df.drop(columns='listed_date')
    #split zipcode to only look at the first 5 numbers, remove anything after dash
    withdash = cars_df['dealer_zip'].str.contains('-') == True
    split = cars_df[withdash]['dealer_zip'].str.split('-', n =1, expand = True)
    cars_df.loc[withdash, 'dealer_zip'] = split[0]
    cars_df = cars_df.drop(columns=['listing_id'])
    split6 = cars_df['trimId'].str.split('t', n =1, expand = True)
    cars_df['trimId'] = split6[1]
    cars_df['trimId'] = pd.to_numeric(cars_df['trimId'], errors='coerce')
    cars_df = cars_df.drop(columns='trim_name')
    cars_df = cars_df.drop(columns='wheel_system_display')
    cars_df = cars_df.drop(columns='transmission')
    split7 = cars_df['maximum_seating'].str.split(' ', n =1, expand = True)
    cars_df['maximum_seating'] = split7[0]
    cars_df['maximum_seating'] = pd.to_numeric(cars_df['maximum_seating'], errors='coerce')

    #engine_cylinders and engine_type are the same column, drop #engine_cylinders
    cars_df = cars_df.drop(columns='engine_cylinders')
    
    return cars_df
    
    