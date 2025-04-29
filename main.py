
# Import necessary libraries
import nest_asyncio
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
#from pyngrok import ngrok
import io

nest_asyncio.apply()

app = FastAPI()

model = EfficientNetV2B0(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_bytes):
    try:
        img_bytes_io = io.BytesIO(img_bytes)

        img = image.load_img(img_bytes_io, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array)
        return features
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

threshold = 0.5

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        features1 = extract_features(await file1.read())
        features2 = extract_features(await file2.read())

        if features1 is None or features2 is None:
            return JSONResponse(content={"error": "Failed to extract features from one or both images."}, status_code=500)

        similarity = cosine_similarity(features1, features2)[0][0]

        print(f"Cosine similarity: {similarity}")

        if similarity > threshold:
            return JSONResponse(content={"similar": True})
        else:
            return JSONResponse(content={"similar": False})

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

#public_url = ngrok.connect(8000)
#print("Public URL:", public_url)

port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)