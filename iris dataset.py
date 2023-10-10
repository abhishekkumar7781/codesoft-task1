#!/usr/bin/env python
# coding: utf-8

# ### Import Modules
# 

# In[30]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Loading dataset

# In[31]:


df = pd.read_csv('Iris.csv')
df.head()


# In[32]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[33]:


# to display stats about data
df.describe()


# In[34]:


# to basic info about datatype
df.info()


# In[35]:


# to display no. of samples on each class
df['Species'].value_counts()


# ### Preprocessing the dataset
# 

# In[36]:


# check for null values
df.isnull().sum()


# ### Exploratory Data Analysis

# In[37]:


# histograms
df['SepalLengthCm'].hist()


# In[38]:


df['SepalWidthCm'].hist()


# In[39]:


df['PetalLengthCm'].hist()


# In[40]:


df['PetalWidthCm'].hist()


# In[41]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[42]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[43]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[44]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[45]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# ### Coorelation Matrix

# In[46]:


df.corr()


# In[47]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[48]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[49]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# ### Model Training

# In[50]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[51]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[52]:


# model training
model.fit(x_train, y_train)


# In[53]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

