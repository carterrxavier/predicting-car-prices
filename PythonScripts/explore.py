import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools

def get_distribution(df):
    for i in df.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data = df, x=i)
    plt.show()
    
    
    
def graph_to_target(df, target):
    for i in df: 
        if(df[i].dtypes != 'object'):
            plt.figure(figsize=(7,5))
            sns.regplot(data=df, x=i, y = target,line_kws={"color":"black"},scatter_kws={"color":'pink','alpha':0.5})
            plt.show()
        else:
            plt.figure(figsize=(7,5))
            sns.scatterplot(data = df, x= i, y= target)
            plt.show()
    
def get_heatmap(df, target):
    '''
    This method will return a heatmap of all variables and there relation to churn
    '''
    plt.figure(figsize=(15,12))
    color = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending=False), annot=True, cmap=color)
    heatmap.set_title('Feautures Correlating to {}'.format(target))
    plt.show()
    return heatmap


    
def plot_variable_pairs(df, cont_vars = 2):
    combos = itertools.combinations(df,cont_vars)
    for i in combos:
        plt.figure(figsize=(20,12))
        sns.regplot(data=df, x=i[0], y =i[1],line_kws={"color":"black"},scatter_kws={"color":'pink','alpha':0.5})
        plt.show()
        



def plot_cat_and_cont(cat_var, con_var, df):
    for i in cat_var:
        for j in con_var:
            plt.figure(figsize=(20,12))
            plt.subplot(131)
            sns.swarmplot(x=i, y=j, data=df)
            plt.subplot(132)
            sns.boxplot(x=i, y=j, data=df)
            plt.subplot(133)
            sns.barplot(x=i, y=j, data=df)
            plt.show()
        

