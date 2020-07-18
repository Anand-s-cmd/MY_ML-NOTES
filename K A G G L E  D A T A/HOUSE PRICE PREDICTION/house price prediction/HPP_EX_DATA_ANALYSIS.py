""" -*- coding: utf-8 -*-   """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

pd.pandas.set_option('display.max_columns',None)

dataset=pd.read_csv("C:\\EXCELR\\NOTES WRITTEN\\K A G G L E  D A T A\\house-prices-advanced-regression-techniques\\train.csv")
dataset.shape#(1460, 81)
dataset.head()
"""
In Data Analysis We will Analyze To Find out the below stuff
    Missing Values
    All The Numerical Variables
    Distribution of the Numerical Variables
    Categorical Variables
    Cardinality of Categorical Variables
    Outliers
    Relationship between independent and dependent feature(SalePrice)
"""

"""Missing Values"""
feature_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for feature in feature_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(),4), ' % missing values')
"""
LotFrontage 0.1774  % missing values
Alley 0.9377  % missing values
MasVnrType 0.0055  % missing values
MasVnrArea 0.0055  % missing values
BsmtQual 0.0253  % missing values
BsmtCond 0.0253  % missing values
BsmtExposure 0.026  % missing values
BsmtFinType1 0.0253  % missing values
BsmtFinType2 0.026  % missing values
FireplaceQu 0.4726  % missing values
GarageType 0.0555  % missing values
GarageYrBlt 0.0555  % missing values
GarageFinish 0.0555  % missing values
GarageQual 0.0555  % missing values
GarageCond 0.0555  % missing values
PoolQC 0.9952  % missing values
Fence 0.8075  % missing values
MiscFeature 0.963  % missing values
"""

"""HERE WE ARE GOING TO CHECK THE RELATIONSHIP BETWEEN THE TARGET VARIABLE AND THE MISSING VALUES VARIABLES"""
for feature in feature_with_na:
    data=dataset.copy()
    #lets make a variable that indicates 1 if the record has missing value or 0
    data[feature] = np.where(data[feature].isnull(), 1,0)
    
"""let's calculate the mean SalePrice where the information is missing or present"""
    #print(data.groupby(feature)['SalePrice'].median())
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    
"""     HERE IT SHOWS THAT SEE FOR EXAMPLE LotFrontage

        LotFrontage   SalePrice
        111           1000   ---> 0
        NA            1005   ---> 1
        NA            1010   ---> 1
        222           999    ---> 0
        NA            1015   ---> 1
        444           800    ---> 0
        
        BECOMES A GROUP LIKE NA AND OTHERS VALUES AS 1 AND 0
        THEN---->
        
        NA            1005   ---> 1
        NA            1010   ---> 1
        NA            1015   ---> 1
        
        AND-----> 
        
        111           1000   ---> 0
        222            999   ---> 0
        444            800   ---> 0
        
        THATS WHY IT SHOWING MORE VALUES ON GRAPH FOR 1's AND LESS FOR 0's
        AGAINST THE "SalePrice" VALUES 
"""

"""
Here With the relation between the missing values and the dependent variable is clearly visible.
So We need to replace these nan values with something meaningful which we will do in the 
Feature Engineering section
"""

""" From the above dataset some of the features like Id is not required """
print("Id of Houses {}".format(len(dataset.Id)))#Id of Houses 1460


        
"""Numerical Variables"""

# list of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print("Number of numerical features :", len(numerical_features))#Number of numerical features : 38

# visualise the numerical variables
dataset[numerical_features].head(5)

""" Temporal Variables(Eg: Datetime Variables) """
## list of variables that contain year information

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
""" ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'] """

# let's explore the content of these year variables

for feature in year_feature:
    print(feature, dataset[feature].unique())

## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median of House Price')
plt.title("House Price VS Year_Sold")


""" ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'] """

"""data['YearBuilt']=data['YrSold']-data['YearBuilt']"""
"""
1970 = 1980 - 1970 = 10
HERE IT SHOWS THAT YEAR BUILT IS 10 YEARS OLD SO WHAT HAPPENS IS
FROM 0 TO 10 YEARS IT SHOWS US SALEPROCE FROM 0 TO 10 NOW SAME IT HAPPENS FOR OTHER FEATURES LIKE
YEAR OF MODIFICATION AND GARAGE BUILT YEAR

'YearRemodAdd', 'GarageYrBlt'

"""

## Here we will compare the difference between All years feature with SalePrice
for feature in year_feature:
    if feature!= 'YrSold':
        data=dataset.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature] = data['YrSold'] - data[feature]
    
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
    
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature = [feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['ID']]
print("Discrete features count: {}".format(len(discrete_feature)))#Discrete Variables Count: 17
"""
['MSSubClass',
 'OverallQual',
 'OverallCond',
 'LowQualFinSF',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 '3SsnPorch',
 'PoolArea',
 'MiscVal',
 'MoSold']
"""    
## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
   
    
"""Continuous Variable"""

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("continuous features leangth: {}".format(len(continuous_feature)))#16

### Lets analyse the continuous values by creating histograms to understand the distributio

for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature);plt.ylabel("count");plt.title(feature);
    plt.show()
    

""" Exploratory Data Analysis Part 2 """
## We will be using logarithmic transformation

for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature);plt.ylabel('SalePrice');
        plt.title('feature')
        plt.show()
"""
        data[feature].hist(bins=25)
        plt.xlabel(feature);plt.ylabel("count");plt.title(feature);
        plt.show()
        
WHEN EXECUTE ABOVE CODE THEN FEATURE WE CAN SEE FROM HISTOGRAM THAT
THEY ARE SOMEHOW FOLLOWING G NORMAL DISTRUBUTION
ITS GOOD
"""
    
""" Outliers """

for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        #plt.xlabel(feature);
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        

""" Categorical Variables """

categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
dataset[categorical_features].head()

""" NOW CHECKING EACH CATEGORICAL VARIABLE SUB CATEGORIES OR LENGTH """
for feature in categorical_features:
    print('FEATURE {} & NO. OF CATEGORIES {}'.format(feature,len(dataset[feature].unique())))

for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature);plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    