import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import uvicorn
import cv2

from config import *

app = FastAPI()
model = load_model(model_save_path)


def detect(model, preprocessed_image):
    output = model.predict(preprocessed_image)[0]

    predictions = {}
    for index in label_decoder.keys():
        predictions[label_decoder[index]] = float(output[index])

    return predictions


def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image


def prepare_image(image: Image.Image):
    image = np.array(image)

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = cv2.resize(image, (256, 256))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image = await file.read()
    image = read_imagefile(image)
    image = prepare_image(image)
    predictions = detect(model, image)
    return {"predictions": predictions}


@app.get("/")
async def public_html():
    return FileResponse(path='./public/index.html')


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
