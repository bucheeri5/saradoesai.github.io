#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Week 3: Exploring Data in Multiple Ways

# In[4]:


from sklearn import datasets


# In[5]:


dir(datasets)


# I chose make_moons & load_sample_Image

# In[6]:


Moons=datasets.make_moons()
Sample_Images = datasets.load_sample_image
Wine_Data = datasets.load_wine()


# In[7]:


print(Moons)


# In[8]:


Sample_Images


# In[9]:


print(Wine_Data.DESCR)


# In[10]:


from sklearn import datasets
import pandas

wine_data = datasets.load_wine()
wine_dataframe = pandas.DataFrame(data=wine_data["data"], columns = wine_data["feature_names"])


# In[11]:


wine_dataframe.head()


# In[12]:


wine_dataframe.describe()

