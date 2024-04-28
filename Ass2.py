# runs perfectly
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

def RemoveOutlier(df, var):
    Q1 = df[var].quantile(0.25) 
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5*IQR 
    high = Q3 + 1.5*IQR
    df = df[(df[var]>=low) & (df[var] <= high)] 
    return(df)

def DisplayOutliers(df, message):
    fig, axes = plt.subplots(2,2)
    fig.suptitle(message)
    sns.boxplot(data=df, x='raisedhands', ax=axes[0,0])
    sns.boxplot(data=df, x='VisITedResources', ax=axes[0,1])
    sns.boxplot(data=df, x='AnnouncementsView', ax=axes[1,0])
    sns.boxplot(data=df, x='Discussion', ax=axes[1,1])
    fig.tight_layout()
    plt.show()

df = pd.read_csv('student_data.csv')
print('student academic performance dataset is loaded')

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
print('statistical information of numerical columns : \n', df.describe())

# see there any missing values
print('Null values are : \n',df.isnull().sum())

# handling outliers 
DisplayOutliers(df, 'Before removing outliers ')
df=RemoveOutlier(df, 'raisedhands')
df=RemoveOutlier(df, 'VisITedResources')
df=RemoveOutlier(df, 'AnnouncementsView')
df=RemoveOutlier(df, 'Discussion')
DisplayOutliers(df, 'After removing outliers ')

# conversionn of categorical to quantitative (encoding)
df['gender']=df['gender'].astype('category')
df['gender']=df['gender'].cat.codes
print('data types of gender after label encoding = ',df.dtypes['gender'])
print('gender values: ', df['gender'].unique())

sns.boxplot(data=df, x='gender', y='raisedhands', hue='gender')
plt.title('boxplot with 2 variables gender and raisedhands')
plt.show()

sns.boxplot(data=df, x='NationalITy', y='Discussion', hue='gender')
plt.title('boxplot with 3 variables gender, nationality and discussion')
plt.show()

print('relationship between variables using scatterplot: ')
sns.scatterplot(data=df, x='raisedhands', y='VisITedResources')
plt.title('scatterplot for raisedhands, VisITedResources')
plt.show()