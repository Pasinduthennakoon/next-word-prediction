import numpy as np
import tensorflow as tf

with open("./cricket.txt", "r", encoding="utf-8") as file:
    cricket_history = file.read()

tokernizer = tf.keras.preprocessing.text.Tokenizer()
tokernizer.fit_on_texts([cricket_history])

input_sequences = []
for sentence in cricket_history.split("\n"):
    print(sentence)
    token_list = tokernizer.texts_to_sequences([sentence])[0]
    print(token_list)