from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Define the prediction endpoint
@app.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((256, 256))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        preds = model.predict(image)
        labels = ['Healthy', 'Powdery', 'Rust']
        preds_class = np.argmax(preds)
        preds_label = labels[preds_class]

        return JSONResponse(content={
            "Plant Condition": preds_label,
            "Confidence Percentage": int(round(float(preds[0][preds_class]) * 100))
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



