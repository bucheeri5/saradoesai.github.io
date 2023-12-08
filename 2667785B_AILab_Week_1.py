#!/usr/bin/env python
# coding: utf-8

# <a href src="https://github.com/bucheeri5/saradoesai.github.io">Heres a link to my repositry</a>

# # Week 1: Getting started with Anaconda, Jupyter Notebook, and Python
# Exercises to familiarize myself with Jupyter Notebook and its relationship with Python

# a) Why you chose to join this course – for, motivation, vision, aspiration?
# 
# I am interested in humanities computing and using technology to enhance information processes in the arts. 
# 
# b) Prior experience, if any, you have with AI and/or Python
# 
# I have prior experience with Python and some APIs
# 
# c) What you expect to learn from the course 
# 
# To be proficient in Python,
# 
# To understand the principles of artificial intellegence,
# 
# To be able to apply my skills in making basic AI related projects. 

# In[1]:


print ("Hello, World!")
message = ("Hello, its me Sara!")
print (message)
print (message + message)
print (message*3)
print (message [0])
print (message [5])


# In[2]:


from IPython.display import *


# In[3]:


YouTubeVideo("dAwLMS8fgoA") 


# In[ ]:


import webbrowser
import requests

print("Shall we hunt down an old website?")
site = input("Type a website URL: ")
era = input("Type year, month, and date, e.g., 20150613: ")
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era)
response = requests.get(url)
data = response.json()
try:
       old_site = data["archived_snapshots"]["closest"]["url"]
       print("Found this copy: ", old_site)
       print("It should appear in your browser.")
except:
       webbrowser.open(old_site)
       print("Sorry, could not find the site.")

