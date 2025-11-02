import tensorflow as tf
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model('model.h5')

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Input text
input_text = "cricket begin"

# Convert to sequence using the trained tokenizer
token_list = tokenizer.texts_to_sequences([input_text])[0]

print("Tokenized input:", token_list)
