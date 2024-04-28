# runs perfectly

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

df=pd.read_csv('iris.csv') 
print(df)

column_name=['sepal.length','sepal.width','petal.length','petal.width', 'variety'] 
df.columns=column_name
df.head() 
print(df.info())

# #Histrogram with 1-variable
fig, axes = plt.subplots(2,2)

sns.histplot(df['sepal.length'], bins=5,ax=axes[0,0]) 
# plt.show()
sns.histplot(df['sepal.width'], bins=5,ax=axes[0,1])
# plt.show()
sns.histplot(df['petal.length'], bins=5,ax=axes[1,0]) 
# plt.show()
sns.histplot(df['petal.width'], bins=5,ax=axes[1,1])
plt.show()

#Histogram with 2-variable
sns.histplot(data=df, x='sepal.length', hue='variety',multiple='dodge')
plt.show()
sns.histplot(data=df, x='sepal.width', hue='variety',multiple='dodge')
plt.show()
sns.histplot(data=df, x='petal.length', hue='variety',multiple='dodge')
plt.show()
sns.histplot(data=df, x='petal.width', hue='variety',multiple='dodge')
plt.show()

# boxplot of 1 variable
sns.boxplot(data=df, x='sepal.length')
plt.show()
sns.boxplot(data=df, x='sepal.width')
plt.show()
sns.boxplot(data=df, x='petal.length')
plt.show()
sns.boxplot(data=df, x='petal.width')
plt.show()

# boxplot of 2 variable
sns.boxplot(data=df, x='sepal.length',y='variety',hue='variety')
plt.show()
sns.boxplot(data=df, x='sepal.width',y='variety',hue='variety')
plt.show()
sns.boxplot(data=df, x='petal.length',y='variety',hue='variety')
plt.show()
sns.boxplot(data=df, x='petal.width',y='variety',hue='variety')
plt.show()


