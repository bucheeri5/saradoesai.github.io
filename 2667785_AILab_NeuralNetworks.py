#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Generating Text with Neural Networks
# 

# This code employs neural networks to generate text based on Shakespeare literature. It uses TensorFlow, which is a Python library for building neural networks. This model shows how the humanities can be engaged within LLMs, beyond general generative AI. Yet, building specialised data sets and categorising them for machine learning is challenging because it requires significant resources. Notably, this project required significant computing energy and time to run. This means it needs computers with higher capacity. This project reduced the training data due to the capacity of my personal computer, and this notably affected the data as some of the outputs were incoherent and others were gibberish. The implications of these points are important because, at the current stage of humanities and low funding, proceeding with highly specialised projects like this on a much larger scale might require more resources, which could be ambitious within the humanities. Neural networks are also slightly harder to interpret because they require advanced code, so this project added comments to allow a wider audience to engage with the process. However, it also points out how humanities students should be equipped with digital literacy skills in order to integrate with modern-day technologies.

# # Getting the Data

# In[1]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url) 
with open(filepath) as f: 
    shakespeare_text = f.read() 


# In[2]:


print(shakespeare_text[:80]) # This will print the first 80 characters of the data


# # Preparing the Data

# The code converts Shakespeare's dataset into a format understandable by a computer, enabling computational analysis. It helps in tasks like understanding patterns, sentiment analysis, or building models for various natural language processing applications. This section of code snippet sets the length of sequences to 100 and initializes a random seed for reproducibility using TensorFlow. Following that it creates three datasets—training, validation, and test sets—using the to_dataset function. The training set is derived from the first 125,000 elements of the encoded data, with sequences of the specified length, and is shuffled for randomness. Imagine the text as one long story, and this code is organizing it into three parts: a training section to teach the program, a validation section to check its learning, and a test section to see how well it can apply what it learned. The length of each piece of the story that the program sees at once is set to 100 characters, like a small snippet.  The validation set comprises sequences from 125,000 to 132,500 of the encoded data. Lastly, the test set is formed from elements starting from the 132,500th position in the encoded data, all with sequences of the specified length. These datasets are likely intended for training and evaluating a machine learning model on the processed Shakespearean text data. The goal is to train a computer model to understand and generate text in a way that resembles Shakespeare's writing style. Preparing the data is the most time consuming part in terms of the process but the most important! This is what our neural network learns from, so we need to figure out what we are feeding it.
# 
# 
# 
# 
# 

# In[3]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower") #This line creates a layer that converts text into numerical vectors. It takes two arguments: split and standardize. split tells the layer to split the text into individual characters, and standardize tells it to convert all letters to lowercase.
text_vec_layer.adapt([shakespeare_text]) # This line adapts the layer to the text data. It's like telling the layer, "Hey, I'm going to give you some text to work with. Get ready!"
encoded = text_vec_layer([shakespeare_text])[0] # This line applies the layer to the text data and gets the output. The [0] at the end tells it to return the first element of the output, as python code starts counting from 0 instead of 1, which represents the encoded text.


# In[4]:


print(text_vec_layer([shakespeare_text])) #this code displays the numerical representation of the text after the processing performed by the text vectorization layer.


# In[5]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# In[6]:


print(n_tokens, dataset_size) # Print the number of distinct characters and dataset size which is the number of characters


# In[7]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# In[8]:


length = 100
tf.random.set_seed(42) # Set random seed for reproducibility

train_set = to_dataset(encoded[:125_000:], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[125_000:132_500], length=length)
test_set = to_dataset(encoded[132_500:], length=length)


# # Building and Training the Model

# In this section, the goal is to make the model learn from the organised Shakespearean text selected and snipped in the previous step. The code establishes a random seed to make sure that the results are reproducible, meaning if I were to run the programme multiple times, I'd get the same outcomes. The programme is structured in layers. First, it sets up a layer to convert the individual characters into numerical representations. Then, it uses a type of layer called GRU to understand patterns in the data, and finally, it has a layer that predicts the next character in the sequence. The model is trained using a specific loss function and an optimisation algorithm. The loss function guides the model during training by penalizing it when the predicted probabilities diverge and stray away from the characters observed in the training data that we specified earlier on. Minimising this loss is the objective of the training process, as it leads the model to make more accurate predictions over time. The number of epochs is specified by the epochs=10 parameter in the model.fit function. In this case, the model will undergo training for ten epochs, meaning it will iterate through the entire train_set dataset ten times. Adjusting the number of epochs is a common hyperparameter tuning step in machine learning, and the optimal value often depends on the specific characteristics of the dataset and the complexity and size of the model. Increasing the data size and epochs will optimise the results, but that also requires more computing power and processing time. The choice is to be made by the developers, depending on how accurate the outcomes we want them to be! If you are really keen on Shakespear and are offended when people misquote him, then you might want to increase the data size and epochs. I am not, so I reduced the data. In this code, there's a checkpoint set up to save the best version of the model during training, and the training history is stored for later analysis. This entire process is aimed at creating a model that can generate text in a way that mimics Shakespeare's writing style. This is a very time-consuming process to run, but it is basically waiting for the computer to marinate the data and analyse it based on what we told it to. Here, we tell it what we have, and we sit back and relax till it loads.
# 
# 
# 
# 
# 

# In[9]:


# Set a random seed to ensure reproducibility of results
tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
# Compile the model with specific settings
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
# Model checkpoint callback
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[10]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# Arguably, this is the most fun part of this process! This is because we get to test whether the text aligns with Shakespear.

# In[11]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[12]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[13]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[14]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[15]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[16]:


print(extend_text("To be or not to be", temperature=0.01))


# The statement is not a very coherent or clear statement, as it grammatically incomplete. It is difficult to understand what it means. 
# 
# 

# In[17]:


print(extend_text("To be or not to be", temperature=1))


# This shows some elements from Shakespeare's literature. I believe that the words "basbere" and "dishacond" is mimicking how Shakespeare was inventive with language. He was well-known for coining new terms that were derived from existing ones, regardless if the model was really poor at this. However, it is not comprehensible.
# 
# 

# In[18]:


print(extend_text("To be or not to be", temperature=100))


# Fully gibberish- very poor quality.

# Overall, this model is proving itself poor in the quality. One of the most reasonable explanations for this is that I reduced the size of the data to have a quicker loading time. When I compared outcome with my peer support group, they did have more coherent data in comparison to mine, which explains my outcomes. However, it is admirable that the dataset is only based on Shakespear and was able to produce semi-impressive results with my peers who ran all the data available. However, as mentioned in the beginning this raises issues about how viable it is within the humanities. Perhaps cultural items that are generally percieved as more prestigious and favorable by other people will have more datasets, but works from lesser known artists might not be pursued for such datasets. It would be interesting to see the application of this within exhibitions, such as robots that pretend to be certain figures from history that have been mimicked through generative AI. Once again, this would require extensive resources!
