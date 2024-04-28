import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read dataset
df=pd.read_csv('placement_data.csv')
print('Placement dataset successfully loaded into dataframe...')

# display information of dataset
print('Information of dataset: \n',df.info)
print('shape of dataset (row x column):', df.shape)
print('columns name: ',df.columns)
print('total elements in dataset: ', df.size)
print('datatype of attributes (columns) :', df.dtypes)
print('first 5 rows:\n', df.head().T)
print('last 5 rows:\n', df.tail().T)
print('any 5 rows:\n', df.sample(5).T)

# display statistical information of dataset
print('statistical information of numerical columns: \n', df.describe())

# display null values
print('total number of null values in dataset : \n', df.isna().sum())

# data type conversion
print('converting data type of variables : \n')
df['sl_no']=df['sl_no'].astype('int8')
print('check datatype of sl_no : ', df.dtypes)
df['ssc_p']=df['ssc_p'].astype('int8')
print('check datatype of ssc_p : ', df.dtypes)

# label encoding conversion of categorical to quantitative
print('encoding using label encoding (cat codes) : ')
df['gender']=df['gender'].astype('category')
print('data types of gender: ',df.dtypes['gender'])
df['gender']=df['gender'].cat.codes
print('data types of gender after label encoding = ', df.dtypes['gender'])
print('gender values: ', df['gender'].unique())

# normalization
print('normalization using min-max feature scaling: ')
df['salary']=(df['salary']-df['salary'].min())/(df['salary'].max()-df['salary'].min())
print(df.head().T)
