def RemoveOutliers(df, var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def DisplayOutliers(df, msg):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(msg)
    sns.boxplot(data = df, x = 'sepal.length', ax = axes[0, 0])
    sns.boxplot(data = df, x = 'sepal.width', ax = axes[0, 1])
    sns.boxplot(data = df, x = 'petal.length', ax = axes[1, 0])
    sns.boxplot(data = df, x = 'petal.width', ax = axes[1, 1])
    fig.tight_layout()
    plt.show()


#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv('iris.csv')
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

print('Finding and removing outliers')
DisplayOutliers(df, 'Before removing outliers')
df = RemoveOutliers(df, 'sepal.length')
df = RemoveOutliers(df, 'sepal.width')
df = RemoveOutliers(df, 'petal.length')
df = RemoveOutliers(df, 'petal.width')
DisplayOutliers(df, 'After removing outliers')

#Encoding of output variable
df['variety'] = df['variety'].astype('category')
df['variety'] = df['variety'].cat.codes
print('The alues associated with the variety will be: ', df['variety'].unique())

#Find correlation matrix
print('Finding correlation matrix using heatmap: ')
sns.heatmap(df.corr(), annot=True)
plt.show()

#Split the data into inputs and outputs
x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values

#Training and testing data
from sklearn.model_selection import train_test_split

#Assign test data size 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Normalization of input data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#Apply Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Display classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Display confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix\n', cm)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=.3, cmap="Blues")
plt.show()


