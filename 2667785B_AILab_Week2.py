#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Week 2: Exploring Data in Multiple Ways

# In[1]:


from IPython.display import Image


# In[2]:


Image ("picture1.jpg")


# In[3]:


from IPython.display import Audio


# In[4]:


Audio ("audio1.mid")


# In[5]:


Audio ("audio2.ogg")
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license. 
#You are free: to share – to copy, distribute and transmit the workto remix – to adapt the work
#Under the following conditions: 
#atribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
#ou may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.hare alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# The .mid file doesnt work because the file type is Musical Instrument Digital Interface which doesnt carry the audio, and is not supported by the browser, hence why it does not play, unlike the .ogg file. 

# In[6]:


from matplotlib import pyplot 
test_picture = pyplot.imread("picture1.jpg")
print ("Numpy array of the image is: " , test_picture)
pyplot.imshow(test_picture)
test_picture_filtered = 2*test_picture/3
pyplot.imshow(test_picture_filtered)


# In[ ]:




