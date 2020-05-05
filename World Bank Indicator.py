#!/usr/bin/env python
# coding: utf-8

# ***
# 
# <h1> World Bank Development Indicators and Income Group Estimation </h1>
# 
# 
# A data set of World Bank development indicators. It contains development indicators in many different headings according to years, countries and regions. We will estimate the income group of income countries using 2012 data. In our forecasting model, the per capita income of countries, inflation rate, export rate, import rate, population growth rate, agriculture value added, industry value added, manufacturing value added and income group are used. Since the income group is a categorical variable, the Logistic Regression model was used.
# 

# <h2> Data Preparation and Exploration</h2>

# In[28]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('Indicators.csv')
mf = pd.read_csv('Country.csv')

# Merging data sets
cf = pd.merge(df,mf[["CountryCode","IncomeGroup"]], on="CountryCode")


# In[29]:


cf.set_index("CountryName",inplace=True)

tf = cf[cf["Year"]==2012]

# Delete Nan values from income group
tf.dropna(subset=["IncomeGroup"], inplace=True)
tf.head()


# In[30]:


# Removing variables that we will not use from the dataset
tf.drop(['CountryCode','IndicatorCode','Year'], axis=1,inplace=True)


# In[31]:


# Extracting the variables I will use from the dataset
gdp = tf[(tf["IndicatorName"]=="GDP per capita (current US$)")]
inflation = tf[(tf["IndicatorName"]=="Inflation, GDP deflator (annual %)")]
export = tf[(tf["IndicatorName"]=="Exports of goods and services (% of GDP)")]
imports = tf[(tf["IndicatorName"]=="Imports of goods and services (% of GDP)")]
population = tf[(tf["IndicatorName"]=="Population growth (annual %)")]
manufac = tf[(tf["IndicatorName"]=="Manufacturing, value added (% of GDP)")]
industry = tf[(tf["IndicatorName"]=="Industry, value added (% of GDP)")]
agriculture = tf[(tf["IndicatorName"]=="Agriculture, value added (% of GDP)")]


# In[32]:


data = {'gdp': gdp.Value,'inflation': inflation.Value,'export':export.Value,'imports':imports.Value,'population':population.Value,
     'agriculture':agriculture.Value,
     'industry':industry.Value,
     'manufac':manufac.Value}

variables = pd.DataFrame(data=data)


# In[10]:


variables["IncomeGroup"]=population["IncomeGroup"]

variables.dropna(axis=0,inplace=True)

variables.head()


# In[11]:


# Categorical Feature Conversion 
X = variables.iloc[:,0:-1].values
X.shape


# In[12]:


y = variables.iloc[:,-1].values
y.shape


# <h2> Feature Scaling </h2>

# In[13]:


# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[14]:


# Label Encoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# <h2> Splitting the data into training and test sets </h2>
# 

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# <h2> Modeling and Training </h2>

# In[16]:


classifier = LogisticRegression(random_state=25, solver='lbfgs')
classifier.fit(X_train, y_train)


# In[33]:


y_pred = classifier.predict(X_test)


# <h2> Making estimation results dataframe from numpy.ndarray </h2>

# In[34]:


predictions = pd.DataFrame(data=y_pred,    # values
                index=range(len(y_pred)),    # 1st column as index
                   columns=['y_pred'])  # 1st row as the column names

# Sadece y_pred'den oluşan df'e test(gerçek) y_test'i sütun olarak ekleme
predictions['y_test'] = y_test
predictions.head()


# <h2> Model Evaluation </h2>

# In[19]:


# Confusion Matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)


# <h2> Classification Performance Evaluation (Accuracy) </h2>

# In[21]:


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: % {:10.2f}".format(accuracy*100)) 


# In[22]:


print(classification_report(y_test, y_pred))


# In[23]:


sns_plot = sns.pairplot(variables,hue="IncomeGroup",size=1.5)


# In[24]:


sns.jointplot(x="gdp",y="inflation",data=variables,kind="reg")


# In[25]:


sns.set(style="darkgrid",font_scale=1.5)

f, axes = plt.subplots(4,2,figsize=(16,20))

sns.distplot(variables["gdp"],color="#d7191c",ax=axes[0,0])

sns.distplot(variables["inflation"],color="#fdae61",ax=axes[0,1])

sns.distplot(variables["export"],color="#abd9e9",ax=axes[1,0])

sns.distplot(variables["imports"],color="#2c7bb6",ax=axes[1,1])

sns.distplot(variables["population"],color="#018571",ax=axes[2,0])

sns.distplot(variables["agriculture"],color="r",ax=axes[2,1])

sns.distplot(variables["industry"],color="g",ax=axes[3,0])

sns.distplot(variables["manufac"],color="#a6611a",ax=axes[3,1])


# In[ ]:




