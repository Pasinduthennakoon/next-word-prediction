import tensorflow as tf
import numpy as np
import pickle

input_text = input("enter your input text:")
cnt = int(input("enter your word count:"))

# Load model
model = tf.keras.models.load_model('model.h5')

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Input text
# input_text = "cricket begin"

# Define the maximum length (same as during training)
max_len = model.input_shape[1] + 1  # you can also hardcode 55 if you know it

# Generate 6 words
for i in range(cnt):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_len - 1, padding='pre')

    # Predict next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted_probs, axis=-1)[0]  # get integer id

    # Find the corresponding word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            output_word = word
            break

    # Stop if no word found
    if output_word == "":
        break

    # Append word to input
    input_text += " " + output_word

print("Generated text:", input_text)
