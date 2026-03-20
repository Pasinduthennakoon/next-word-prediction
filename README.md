#  AI Cricket Text Generator (LSTM + FastAPI + Streamlit)

A full-stack AI application that generates cricket-related text using a deep learning model (LSTM).  
The system includes a trained NLP model, a FastAPI backend for inference, and a Streamlit frontend for user interaction.

---

##  Features

-  Next-word prediction using LSTM
-  FastAPI backend for real-time API inference
-  Streamlit frontend for interactive UI
-  Custom NLP model trained on cricket dataset
-  Configurable text generation length
-  Model & tokenizer persistence

---

##  Tech Stack

- Python
- TensorFlow / Keras (LSTM)
- FastAPI
- Streamlit
- NumPy
- Uvicorn
- Requests

---

##  Project Structure
```text
NEXT-WORD-PREDICTION
├──backend
|  ├── cricket.txt
|  ├── app.py
|  ├── train_model.py
|  ├── predict.py
|  └── requirements.txt
├──frontend
|  ├── streamlit_app.py
|  └── requirements.txt
├── .gitignore
├── .gitattribute
└── README.md

```


---

##  Installation

```bash
# Clone repository
git clone https://github.com/Pasinduthennakoon/next-word-prediction.git

cd NEXT-WORD-PREDICTION

# Create virtual environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
cd ../

cd frontend
pip install -r requirements.txt
cd ../

```
---

##  Run

```bash
# Run backend
cd backend
python app.py
# API will run on:
http://127.0.0.1:8000

#run frontend
cd frontend
streamlit run app.py
#streamlit run on:
http://localhost:8501
```
---

## API Endpoint

Request:
```bash
{
  "input_text": "cricket is",
  "word_count": 10
}
```

Response:
```bash
{
  "generated_text": "cricket is a popular sport played worldwide"
}
```

---

## Model Training Pipeline

-  Text preprocessing & tokenization
-  Sequence generation
-  Padding sequences
-  One-hot encoding labels
-  LSTM-based deep learning model

---

## Model Architecture

-  Embedding Layer
-  LSTM Layer(s)
-  Dense (Softmax Output)

---

## Future Improvements

-  Replace LSTM with Transformer (GPT-like)
-  Deploy backend (Render / Railway)
-  Deploy frontend (Streamlit Cloud)
-  Dockerize application
-  Add analytics & logging
