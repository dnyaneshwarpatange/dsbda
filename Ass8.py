# runs perfectly

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read dataset 
df = pd.read_csv('titanic.csv')
print("Titanic dataset loaded successfully")
# print(df.head().T)

# display information of dataset
print('Information of dataset: \n',df.info)
print('shape of dataset (row x column):', df.shape)
print('columns name: ',df.columns)
print('total elements in dataset: ', df.size)
print('datatype of attributes (columns) :', df.dtypes)
print('first 5 rows:\n', df.head().T)
print('last 5 rows:\n', df.tail().T)
print('any 5 rows:\n', df.sample(5).T)

# find missing values
print("Missing values ")
print(df.isnull().sum())

#filling the missing value
df['Age'].fillna(df['Age'].mean(), inplace=True)
#see there any missing values
print('Null values are : \n',df.isnull().sum())

#draw the histogram of 1-variable,2-variable,3-variable

#1-variable = Age and fare

sns.histplot(data=df, x='Age')
plt.show()

sns.histplot(data=df, x='Fare')
plt.show()


# 2-variable 
sns.histplot(data=df, x='Age',hue='Survived',multiple="dodge", shrink=.8)
plt.show()

sns.histplot(data=df, x='Fare', hue="Survived", multiple="dodge", shrink=.8)
plt.show()

sns.histplot(data=df, x='Age', hue="Sex", multiple="dodge", shrink=.8)
plt.show()

sns.histplot(data=df, x='Fare', hue="Sex", multiple="dodge", shrink=.8)
plt.show()
