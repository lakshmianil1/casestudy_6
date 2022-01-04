#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1. Load the dataset into python environment
# 

# In[2]:


data=pd.read_csv('C:/Users/user/Desktop/titanic_dataset.csv')
data.head()


# # 2. Make ‘PassengerId’ as the index column
# 

# In[3]:


data.set_index("PassengerId") 


# # 3. Check the basic details of the datase

# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.describe()


# # 4. Fill in all the missing values present in all the columns in the dataset.

# In[7]:


data.isna().sum()


# In[8]:


data[(data["Age"]<25) & (data["Age"]>35)] 
data


# # 5. Check and handle outliers in at least 3 columns in the dataset

# In[9]:


sns.boxplot(data["Age"])
plt.title("Box plot of Age")
plt.xlabel("Age")


# In[11]:


Q1= 20 
Q3= 38
IQR= Q3-Q1
Upper_limit=Q3+1.5*IQR
Lower_Limit=Q1-1.5*IQR
print ("IQR is ", IQR)
print("Upper limit of normal value is",Upper_limit)
print("Lower limit of normal value is",Lower_Limit)


# In[12]:


outliers=[]
for x in data["Age"]:
    if ((x>Upper_limit) or (x<Lower_Limit)):
        outliers.append(x)
print("outlier data are \n",outliers)

# to find the index of the outlier
ind1= data["Age"]>Upper_limit
print("index of outliers are \n",data.loc[ind1].index)


# In[13]:


data.drop([33, 96, 116, 493, 630, 672, 745, 851], inplace= True)


# In[14]:


sns.boxplot(data["Age"])
plt.title("Box plot of Age")
plt.xlabel("Age")


# In[15]:


sns.boxplot(data["Fare"])
plt.title("Box plot of Fare")
plt.xlabel("Fare")


# In[16]:


Q1=data.Fare.quantile(.25)
Q3=data.Fare.quantile(.75) # instead of copying from describe column we can run a code to find Q1 and Q3
IQR= Q3-Q1
Upper_limit=Q3+1.5*IQR
Lower_Limit=Q1-1.5*IQR
print ("IQR is ", IQR)
print("Upper limit of normal value is",Upper_limit)
print("Lower limit of normal value is",Lower_Limit)


# In[17]:


data[(data.Fare<Lower_Limit)|(data.Fare>Upper_limit)]


# In[18]:


df2= data[(data.Fare>Lower_Limit)&(data.Fare<Upper_limit)]
df2 


# In[19]:


sns.boxplot(df2["Fare"])
plt.title("Box plot of Fare")
plt.xlabel("Fare")


# In[20]:


df3= df2[(df2.Fare>Lower_Limit)&(df2.Fare<Upper_limit)]
sns.boxplot(df3["Fare"])
plt.title("Box plot of Fare")
plt.xlabel("Fare")


# In[21]:


sns.boxplot(data["Pclass"])
plt.title("Box plot of Pclass")
plt.xlabel("Pclass")


# # 6. Do min max scaling on the feature set (Take ‘Survived’ as target)
# 

# In[22]:


data=pd.get_dummies(data)


# In[23]:


data


# In[24]:


X= data.drop(['Survived'], axis=1)
X.describe()


# In[26]:


from sklearn import preprocessing
min_max= preprocessing.MinMaxScaler(feature_range=(0,1))
X=min_max.fit_transform(X)
X=pd.DataFrame(X)
X


# In[27]:


X.describe()


# In[ ]:




