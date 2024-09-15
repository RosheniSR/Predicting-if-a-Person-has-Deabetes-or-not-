#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[6]:


df=pd.read_csv("C:\\Users\\roshe\\Downloads\\diabetes.csv")
df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.Outcome.value_counts()


# In[9]:


X = df.drop('Outcome',axis='columns')
y= df.Outcome


# In[10]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]


# In[12]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,stratify=y,random_state=10)


# In[13]:


X_train.shape


# In[14]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
scores.mean()


# In[15]:


from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(),
                              n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)

bag_model.fit(X_train,y_train)
bag_model.oob_score_


# In[16]:


bag_model.score(X_test,y_test)


# In[17]:


bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(),
                              n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)

scores=cross_val_score(bag_model,X,y,cv=5)
scores.mean()


# In[ ]:




