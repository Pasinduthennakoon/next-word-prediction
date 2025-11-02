import numpy as np
import tensorflow as tf

#load data
with open("./cricket.txt", "r", encoding="utf-8") as file:
    cricket_history = file.read()

#set as numerical
tokernizer = tf.keras.preprocessing.text.Tokenizer()
tokernizer.fit_on_texts([cricket_history])

# add data to sequence
input_sequences = []
for sentence in cricket_history.split("\n"):
    # print(sentence)
    token_list = tokernizer.texts_to_sequences([sentence])[0]
    # print(token_list)
    for i in range (1, len(token_list)):
        sequence = token_list[: i+1]
        input_sequences.append(sequence)

max_len = max([len(input_sequences) for input_sequences in input_sequences])

# add padding
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='pre'))

#feature
x = input_sequences[:, :-1]
#label
y = input_sequences[:, -1]

# print(x[0])
# print(y)

num_classes = len(tokernizer.word_index) + 1
#one hot encoding
y = np.array(tf.keras.utils.to_categorical(y, num_classes=num_classes))


