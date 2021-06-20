import numpy as np
import pandas as pd
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