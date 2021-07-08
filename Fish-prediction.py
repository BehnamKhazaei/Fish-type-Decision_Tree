#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fish_data = pd.read_csv("Fish.csv")
fish_data.head(2)
fish_data.info()
fish_data.groupby('Species').mean()


# In[ ]:





# In[3]:


X = fish_data.drop('Species', axis=1)
y = fish_data['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# In[4]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)


# In[5]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def evaluate_model_performance(y_test, y_pred):
  print(accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))


# In[6]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

evaluate_model_performance(y_test, y_pred)


# In[7]:


import pickle
pickle_out=open("clf.pkl","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()


# In[11]:


clf.predict([[500,42,48,45,7,5]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




