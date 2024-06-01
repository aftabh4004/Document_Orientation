from fastapi import FastAPI, File, UploadFile, Form
from inference import inference
from PIL import Image

app = FastAPI()


@app.post('/')
def perdict_orientaion(
        my_file: UploadFile = File(...),
):
    image = Image.open(my_file.file)
    image = image.convert('RGB')
    result = inference(image)

    return {
        "filename": my_file.filename,
        "orientation": f'{result} degree clockwise'
    }