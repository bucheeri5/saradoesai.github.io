#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Week 3: Exploring Data in Multiple Ways

# In[1]:


from sklearn import datasets # Importing datasets module from scikit-learn (sklearn)


# In[2]:


# Display the databses in the datasets module
dir(datasets) 


# I chose the swiss roll & wine data

# In[3]:


#the swiss roll was hard to figure out. upon constant search about this dataset, i realized i thought it was a recipe, it was not! it is something way cooler!
# I edned up needing to use matplotlib to visualize the data but it is definitely worth it.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Swiss Roll dataset
swiss_roll , color = datasets.make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
# Extract the coordinates from the Swiss Roll data 
x, y, z = swiss_roll.T
# Create a 3D plot to visualize the Swiss Roll
fig = plt.figure(figsize=(8, 8)) # Create a new figure with a size of 8 inches by 8 inches, this also decides the output size
ax = fig.add_subplot(111, projection='3d') # Add a subplot to the figure with a 3D projection because the swiss roll is not 2d
ax.scatter(x, y, z, c=color, cmap=plt.cm.viridis, marker='o') # Create a 3D scatter plot in the subplot
# x, y, and z represent the coordinates of the data points
# c=color: Color the points based on the 'color' variable
# cmap=plt.cm.viridis: Use the 'viridis' colormap for coloring
# marker='o': Use circular markers for the points

# labels for the graph we are about to visualize
ax.set_title("AI For Arts & Humanities: Swiss Roll Dataset")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
#final step: show us the goods!
plt.show()


# In[4]:


# Load the Wine dataset using scikit-learn's 'load_wine' function
Wine_Data = datasets.load_wine()

# Print the description of the Wine dataset
print(Wine_Data.DESCR)


# In[6]:


from sklearn import datasets
# Import the 'pandas' library for data manipulation and analysis
import pandas as pd
# Load the Wine dataset using scikit-learn's 'load_wine' function
wine_data = datasets.load_wine()
# Create a DataFrame using pandas
#Data is the 'data' attribute of the wine_data
#column names are provided by the 'feature_names' attribute of the wine_data
wine_dataframe = pd.DataFrame(data=wine_data["data"], columns=wine_data["feature_names"])


# In[7]:


wine_dataframe.head()


# In[8]:


wine_dataframe.describe()


# In this notebook we got to visualise data in different ways. Learning how to handle data is a very important skills to posses, whether it is at a basic level using excel sheets or making your machine analyse tons of data for you! In an increasingly digital world we need to structure our data to allow us to come to informed conclusion. This is important in environments stressing on evidence-based mangamenet such as healthcare, but it can also help us identify correlations in situations such as musuem visitors and age for example! Visualization and communication is even more imporant, because it surpasses just being able to understand it yourself but also communicating it to an audience.
