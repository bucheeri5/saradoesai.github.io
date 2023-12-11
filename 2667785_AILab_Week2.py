#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Week 2: Exploring Data in Multiple Ways

# In[ ]:


from IPython.display import Image
#this module IPython allows us to import an image, which will be using to di


# In[ ]:


Image ("picture1.jpg")


# What a cute hedgehog!

# In[ ]:


from IPython.display import Audio


# In[ ]:


Audio ("audio1.mid")


# whoops! the audio does not work! why is that? well, if we take a closer look at the audio file format, it is .mid which stands for Musical Instrument Digital Interface. This file type does not carry the audio itself and it is not supported by our browser. Let's try another file type below!

# In[ ]:


Audio ("audio2.ogg")
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license. 
#You are free: to share – to copy, distribute and transmit the workto remix – to adapt the work
#Under the following conditions: 
#atribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
#ou may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.hare alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# Hurray! That works.

# In[ ]:


from matplotlib import pyplot # Import the pyplot module from matplotlib
test_picture = pyplot.imread("picture1.jpg") # Read the image "picture1.jpg" (the previous hedgehog image) and store it as a NumPy array
#NumPy is used to handle image data as arrays efficiently. 
#You can think of arrays like lists, where we can store many variables in a specific sequence at once. In the image, we read them as numbers in terms of pixels.
print ("NumPy array of the image is: " , test_picture) # Print the NumPy array representation of the image 
pyplot.imshow(test_picture) # Display the original image using imshow
test_picture_filtered = 2*test_picture/3 # Apply a simple filter to adjust the brightness by scaling the pixel values
pyplot.imshow(test_picture_filtered) # Display the image we altered  using imshow


# mhmm, perhaps dont use this picture to prove your artistic editing skills.

# In this notebook, we got to display both audio and image files using python. We used different libraries, such as ipython and matplotlib to do that. during the process, we have altered an image by converting it into an array and then altering the brightness. All of this was done mathematically! Perhaps this show that we need to move away from seperating discplines and seeing the beauty and the potential of merging our skills, whether it is in the arts and humanities or STEM.
