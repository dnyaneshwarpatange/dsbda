def RemoveOutliers(df, var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def DisplayOutliers(df, msg):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(msg)
    sns.boxplot(data = df, x = 'Age', ax = axes[0])
    sns.boxplot(data = df, x = 'EstimatedSalary', ax = axes[1])
    fig.tight_layout()
    plt.show()


#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv('Social_Network_Ads.csv')
print('Dataset loaded')

#Display information about dataset
print('Information of dataset:\n', df.info)
print('Shape of dataset (row x column):', df.shape)
print('Column names: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

#Find missing values
print('Missing values:\n', df.isnull().sum())

df.replace(['Male', 'Female'], [1, 0], inplace=True)

#Find correlation matrix
print('Finding correlation matrix using heatmap: ')
sns.heatmap(df.corr(), annot=True)
plt.show()

#Finding and removing outliers
print('Finding and removing outliers: ')
DisplayOutliers(df, 'Before removing outliers:')
df = RemoveOutliers(df, 'Age')
df = RemoveOutliers(df, 'EstimatedSalary')
DisplayOutliers(df, 'After removing outliers:')

#Split the data into inputs and outputs
x = df[['Age', 'EstimatedSalary']]  #input data
y = df['Purchased']                 #output data

#Training and testing data
from sklearn.model_selection import train_test_split

#Assign test data size 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Normalization of input data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#Apply logistic regression model on training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0, solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Display classifiction report
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix\n', cm)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=.3, cmap="Blues")
plt.show()
