from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import os
import uvicorn
from typing import List, Annotated

app = FastAPI()

# Add root route for Render wake-up check
@app.get("/")
def root():
    return {"status": "running"}

# Load the trained fashion classifier model
fashion_model = load_model("fashion_classifier_model.keras")
class_labels = ['bottom', 'dress', 'shoes', 'top']

# Load EfficientNet for feature extraction
feature_model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")

# Extract features for similarity comparison
def extract_features(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_model.predict(img_array)
    return features

# Classify clothing image endpoint
@app.post("/classify_item")
async def classify_item(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = fashion_model.predict(img_array)
    predicted_label = class_labels[np.argmax(preds)]

    return JSONResponse(content={"category": predicted_label})

#  Recommend endpoint (reuses classification + EfficientNet logic)
@app.post("/recommend")
async def recommend_outfit(
    occasion: Annotated[str, Form()],
    pinterest_occasions: Annotated[List[str], Form()],
    pinterest_images: List[UploadFile] = File(...),
    wardrobe_images: List[UploadFile] = File(...)
):
    print("/recommend endpoint hit")

    threshold = 0.3
    matched_outfits = []

    for i, p_img in enumerate(pinterest_images):
        if i >= len(pinterest_occasions):
            continue
        if occasion.strip().lower() != pinterest_occasions[i].strip().lower():
            continue

        p_bytes = await p_img.read()
        p_features = extract_features(p_bytes)
        if p_features is None:
            continue

        matches = []
        for w_img in wardrobe_images:
            print(f"Classifying {w_img.filename}")
            w_bytes = await w_img.read()
            w_features = extract_features(w_bytes)
            if w_features is None:
                continue
            similarity = cosine_similarity(p_features, w_features)[0][0]

            # classify
            w_img_array = image.load_img(io.BytesIO(w_bytes), target_size=(224, 224))
            w_array = image.img_to_array(w_img_array)
            w_array = np.expand_dims(w_array, axis=0) / 255.0
            preds = fashion_model.predict(w_array)
            predicted_label = class_labels[np.argmax(preds)]

            if similarity >= threshold:
                matches.append({
                    "wardrobe_image": w_img.filename,
                    "similarity": float(similarity),
                    "category": predicted_label
                })

        if matches:
            matched_outfits.append({
                "pinterest_image": p_img.filename,
                "recommended_wardrobe": sorted(matches, key=lambda x: x["similarity"], reverse=True)[:3]
            })

    return JSONResponse(content={
        "occasion": occasion,
        "matched_outfits": matched_outfits
    })

# Run locally
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
