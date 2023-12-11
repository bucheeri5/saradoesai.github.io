#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

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
# To be proficient in communicating code and how AI works to a wider audience where digital literacy is becoming increasingly important,
# 
# To understand the principles of artificial intellegence,
# 
# To be able to apply my skills in making basic AI related projects. 

# In[1]:


print ("Hello, World!")
message = ("Hello, its me!")
print (message)
print (message + message)
print (message*3)
print (message [0])
print (message [5])


# In[2]:


from IPython.display import *


# In[3]:


YouTubeVideo("uA70ZGCC1f4") #TED's videos may be used for non-commercial purposes under a Creative Commons License,.


# In[ ]:


# Import the required modules to be able to execute this code
import webbrowser
import requests

# Prompt for users to input a website
print("Shall we hunt down an old website?")
site = input("Type a website URL: ")
era = input("Type year, month, and date, e.g., 20150613: ")

# URL for the Wayback Machine API
#APIs are important in coding, Application Programming Interfaces are sets of rules and tools that allow different software applications to communicate with each other
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era)

# Send a GET request to the Wayback Machine API
response = requests.get(url)

# Parse the JSON response
# This means converting the JSON-formatted data received from the Wayback Machine API into a usable Python data structure.
data = response.json()

# Try to extract the URL of the closest archived snapshot in the wayback machine
try:
    old_site = data["archived_snapshots"]["closest"]["url"]
    print("Found this copy: ", old_site)

    # Open the archived site in the default web browser
    webbrowser.open(old_site)
except:
    # If this does not work, the user will receive following statement
    print("Sorry, could not find the site.")

