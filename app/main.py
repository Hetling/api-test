from turtle import down
from fastapi import FastAPI, File, UploadFile 
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import whisper
import time
import os

model = whisper.load_model('medium')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/file/")
async def create_upload_file(file: UploadFile = File(...)):
    """Return name of file

    Args:
        file (UploadFile, optional): file. Defaults to File(...).

    Returns:
        response: api response containing name of file
    """
    return {"filename": file.filename}

@app.post("/predict/")
async def create_file(file: UploadFile = File(...)):
    """Predicts text from given audio file

    Args:
        file (UploadFile, optional): FastApi filetype or file-like object. Defaults to File(...).

    Returns:
        response : api response containing predicted text and time it took to predict
    """
    s = time.time()
    try:
        temp = NamedTemporaryFile(delete=False)
        with temp as f:
            f.write(file.file.read())
            res = {'result': model.transcribe(temp.name,language='da', fp16=False)['text'], 'time': time.time() - s}
   
    except Exception as e:
        res = {'result': e, 'time': time.time() - s}
    
    finally:
        os.remove(temp.name)
        
        return res
    

@app.get("/example")
def predict():
    """Run example audio file, for

    Returns:
        response: api response containing text and time
    """
    if model == None:
        return {"error": "model not loaded"}
    else:
        s = time.time()
        return {'result': model.transcribe('../app/audio.mp3',language='da', fp16=False)['text'], 'time': time.time() - s}
