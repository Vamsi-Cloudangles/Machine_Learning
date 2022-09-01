#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import pycaret
from pycaret.regression import *
import pickle


# In[28]:


data = pd.read_csv(r"C:\Users\VamsiMuramreddy\Desktop\Insurance-Flask\dataset.csv")


# In[29]:


data


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[31]:


data.duplicated().sum()


# In[32]:


data.drop_duplicates()


# In[33]:


for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])


# In[34]:


data.info()


# In[35]:


data.dtypes


# In[36]:


reg = setup(data = data, target = 'charges')


# In[37]:


best =compare_models()


# In[38]:


best


# In[42]:


best.predict([[50,1,30.970,3,0,1]])


# In[47]:


pickle.dump(best,open('final_model.pkl','wb'))
loaded_model = pickle.load(open('final_model.pkl','rb'))
loaded_model.predict([[50,1,30.970,3,0,1]])


# In[39]:


CBR = create_model(best)


# In[40]:


tuned_CBR = tune_model(best,n_iter = 50)


# In[41]:


plot_model(CBR)


# In[15]:


predict_model(tuned_CBR)


# In[16]:


final_best = finalize_model(best)


# In[17]:


final_best


# In[54]:


with open("model.pkl", 'wb') as f:
     pickle.dump(best,f)


# In[55]:


da = open('model.pkl', 'rb')
lp = pickle.load(da)


# In[56]:


lp.predict([[50,1,30.970,3,0,1]])

