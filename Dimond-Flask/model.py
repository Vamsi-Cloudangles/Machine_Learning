#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from pycaret.regression import *
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[2]:


data = pd.read_csv(r"C:\Users\VamsiMuramreddy\Desktop\Dimond\dataset.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.duplicated().sum()


# In[6]:


data.dtypes


# In[7]:


for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])


# In[8]:


data.dtypes


# In[9]:


data.drop(['SI.NO'], axis =1, inplace =True)


# In[10]:


reg = setup(data = data, target = 'price')


# In[11]:


best = compare_models()


# In[12]:


best


# In[13]:


cbr = create_model(best)


# In[14]:


plot_model(cbr)


# In[16]:


with open("model.pkl", 'wb') as p:
     pickle.dump(best,p)


# In[17]:


pf = open('model.pkl', 'rb')
lp = pickle.load(pf)


# In[18]:


data.head()


# In[19]:


lp.predict([[0.21,3,1,2,59.8,61.0,3.89,3.84,2.31]])


# In[ ]:




