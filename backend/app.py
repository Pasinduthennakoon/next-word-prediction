from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import uvicorn
import numpy as np
import pickle

app = FastAPI()

model = tf.keras.models.load_model('model.h5')

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class TextRequest(BaseModel):
    input_text: str
    word_count: int

@app.post("/generate-text")
def generate_text(request: TextRequest):
    
    input_text = request.input_text
    cnt = request.word_count
    max_len = model.input_shape[1] + 1

    for i in range(cnt):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_len - 1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted_probs, axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output_word = word
                break

        if output_word == "":
            break

        input_text += " " + output_word

    return input_text

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)