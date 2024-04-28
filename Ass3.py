

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('Employee_Salary.csv')
print('Employee_Salary dataset loaded.')

# Display information about dataset
#print('Information of dataset:\n', df.info())
print('Shape of dataset (row x column):', df.shape)
print('Column names:', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# Statistical information of Numerical Columns
print('Statistical information of Numerical Columns: ')
columns = ['Experience_Years', 'Age', 'Salary']
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns', 'Min', 'Max', 'Mean', 'Median', 'STD'))
for column in columns:
    col_min = df[column].min()
    col_max = df[column].max()
    col_mean = df[column].mean()
    col_median = df[column].median()
    col_std = df[column].std()
    print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format(column, col_min, col_max, col_mean, col_median, col_std))

# Groupwise Statistical Summary
print('Groupwise Statistical Summary:')
for column in columns:
    print("{:<20}{:<10}{:<10}{:<20}{:<10}".format('Columns', 'Min', 'Max', 'Mean', 'Median'))
    m1 = df.groupby('Gender')[column].min()
    m2 = df.groupby('Gender')[column].max()
    m3 = df.groupby('Gender')[column].mean()
    m4 = df.groupby('Gender')[column].median()
    s = df.groupby('Gender')[column].std()
    print("{:<20}{:<10}{:<10}{:<20}{:<10}".format('Female', m1['Female'], m2['Female'], m3['Female'], m4['Female']))
    print("{:<20}{:<10}{:<10}{:<20}{:<10}".format('Male', m1['Male'], m2['Male'], m3['Male'], m4['Male']))

# Plotting Groupwise Statistical Information
X = ['min', 'max', 'mean', 'median', 'std']
features = ['Salary', 'Age', 'Experience_Years']
df1 = pd.DataFrame(columns=X)

for var in features:
    df1['min'] = df.groupby('Gender')[var].min()
    df1['max'] = df.groupby('Gender')[var].max()
    df1['mean'] = df.groupby('Gender')[var].mean()
    df1['median'] = df.groupby('Gender')[var].median()
    df1['std'] = df.groupby('Gender')[var].std()

X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, df1.loc['Female'], 0.4, label='Female')
plt.bar(X_axis + 0.2, df1.loc['Male'], 0.4, label='Male')
plt.xticks(X_axis, X)
plt.xlabel('Statistical information')
plt.ylabel('Value')
plt.title('Groupwise Statistical Information of Employee Salary Dataset')
plt.legend()
plt.show()
