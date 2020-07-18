""" -*- coding: utf-8 -*-

We will be performing all the below steps in Feature Engineering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
dataset=pd.read_csv("C:\\EXCELR\\NOTES WRITTEN\\K A G G L E  D A T A\\house-prices-advanced-regression-techniques\\train.csv")
dataset.shape#(1460, 81)

### Always remember there way always be a chance of data leakage so we need to 
#split the data first and then apply feature
## Engineering

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)
X_train.shape,X_test.shape# ((1314, 81), (146, 81))

""" Missing Values """
## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']
for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
    
"""Replace missing value with a new label"""

def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()



""" Now lets check for numerical variables the contains missing values """
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']
# ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

## We will print the numerical nan variables and percentage of missing values
for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.round(dataset[feature].isnull().mean(),4)))

""" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT COING HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!! """
### Replacing the numerical Missing Values
for feature in numerical_with_nan:
## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
dataset[numerical_with_nan].isnull().sum()
    

""" ## Temporal Variables (Date Time Variables)""" 

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dataset[feature]=dataset['YrSold']-dataset[feature]

dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()





















    