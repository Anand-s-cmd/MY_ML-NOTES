""" -*- coding: utf-8 -*-

We will be performing all the below steps in Feature Engineering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
dataset=pd.read_csv("C:\\EXCELR\\NOTES WRITTEN\\K A G G L E  D A T A\\house-prices-advanced-regression-techniques\\test.csv")
dataset.shape#(1460, 80)#SINCE 'SalePrice' TARGET COLUMN IS NOT THERE

### Always remember there way always be a chance of data leakage so we need to 
#split the data first and then apply feature
## Engineering

#from sklearn.model_selection train_test_split
