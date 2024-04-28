# runs perfectly

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df=pd.read_csv('titanic.csv')
print("Titanic dataset loaded successfully...\n")

# display information of dataset
print('Information of dataset: \n',df.info)
print('shape of dataset (row x column):', df.shape)
print('columns name: ',df.columns)
print('total elements in dataset: ', df.size)
print('datatype of attributes (columns) :', df.dtypes)
print('first 5 rows:\n', df.head().T)
print('last 5 rows:\n', df.tail().T)
print('any 5 rows:\n', df.sample(5).T)

# missing values 
print('Missing values ')
print(df.isnull().sum())

# filling the missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)

# see there any missing values
print('Null values are : \n',df.isnull().sum())

# boxplot of 1-variable
sns.boxplot(x=df['Age'])
plt.show()
sns.boxplot(x=df['Fare'])
plt.show()

# two variables
sns.boxplot(data=df,x="Survived",y="Age", hue="Survived")
plt.show()
sns.boxplot(data=df,x="Survived",y="Fare", hue="Survived")
plt.show()

sns.boxplot(data=df, x="Sex", y="Age", hue="Sex")
plt.show()
sns.boxplot(data=df, x="Sex", y="Fare", hue="Sex")
plt.show()



# three variables
sns.boxplot(data=df, x="Sex", y="Age", hue="Survived")
plt.show()
sns.boxplot(data=df, x="Sex", y="Fare", hue="Survived")
plt.show()