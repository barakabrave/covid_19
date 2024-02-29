#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("corona.csv")
df.head()


# In[2]:


print(df.shape)
df.isnull().sum()


# In[3]:


df["Total Vaccinations"]=(df["People Vaccinated"]+df["People Fully Vaccinated"])
df["Total Vaccinations"].isnull().sum()
df["Total Vaccinations"].fillna("0")
df.dropna(inplace=True)
print(df.shape)
df.isnull().sum()


# In[4]:


df["Weekly Cases per Million"]=df["Weekly Cases"]/1000000
df["Weekly Deaths per Million"]=df["Weekly Deaths"]/1000000
df["Total Boosters per Hundred"]=df["Total Boosters"]/100
df["Total Vaccinations per Hundred"]=df["Total Vaccinations"]/100
df["People Vaccinated per Hundred"]=df["People Vaccinated"]/100
df["People Fully Vaccinated per Hundred"]=df["People Fully Vaccinated"]/100
df["Daily Vaccinations per Hundred"]=df["Daily Vaccinations"]/100
df["Daily People Vaccinated per Hundred"]=df["Daily People Vaccinated"]/100
df.head()




# In[5]:


df.drop("Id",axis=1,inplace=True)
df.drop("Location",axis=1,inplace=True)
df.columns


# In[6]:


df.drop("Year",axis=1,inplace=True)
import seaborn as sns
sns.pairplot(data=df, diag_kind='kde')


# In[7]:


sns.heatmap(df[['Weekly Cases', 'Weekly Deaths', 'Total Vaccinations', 'People Vaccinated',
       'People Fully Vaccinated', 'Total Boosters', 'Daily Vaccinations', 'Daily People Vaccinated', "Next Week's Deaths"]].corr(), cmap='Blues', annot=True)
plt.show()


# In[13]:


#removing/reducing the regressor attributes that have less correlation to the response variable
df.drop("Total Boosters",axis=1,inplace=True)
df.drop("People Fully Vaccinated",axis=1,inplace=True)
df.drop("People Vaccinated",axis=1,inplace=True)
df.drop("Total Vaccinations",axis=1,inplace=True)
x=df[['Weekly Cases', 'Weekly Deaths','Daily Vaccinations', 'Daily People Vaccinated']].fillna("0")
y=df["Next Week's Deaths"]
x,y


# In[14]:


x.columns


# In[15]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
#y=sc.transform(y)
x,y


# In[ ]:





# In[16]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(x_train,y_train)
y_pred=model.predict(x_test)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Metrics to evaluate your model 
r2_score(y_test, y_pred)*100, mean_absolute_error(y_test, y_pred), nm.sqrt(mean_squared_error(y_test, y_pred))


# In[17]:


import pickle
pickle_out = open("model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

