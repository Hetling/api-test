from fastapi import FastAPI
import whisper
import torch
import time

model = whisper.load_model('medium')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict():
    if model == None:
        return {"error": "model not loaded"}
    else:
        s = time.time()
        return {'result': model.transcribe('./app/audio.mp3',language='da', fp16=False)['text'], 'time': time.time() - s}
