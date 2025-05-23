from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import numpy as np
import io
import os
import nest_asyncio
import uvicorn
from typing import List, Annotated

nest_asyncio.apply()
app = FastAPI()

# Load EfficientNet for feature extraction
model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")

# Lazy-load Hugging Face transformer model for fashion classification
hf_model = None
hf_extractor = None

def load_hf_model():
    global hf_model, hf_extractor
    if hf_model is None or hf_extractor is None:
        hf_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-fashion_mnist")
        hf_extractor = AutoFeatureExtractor.from_pretrained("nateraw/vit-fashion_mnist")

def extract_features(img_bytes):
    try:
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def classify_image(img_bytes):
    try:
        load_hf_model()  # Lazy-load the model only when needed
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = hf_extractor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        label = hf_model.config.id2label[predicted_idx]
        return label
    except Exception as e:
        print(f"Error classifying image: {e}")
        return "unknown"

@app.post("/classify_item")
async def classify_item(image: UploadFile = File(...)):
    img_bytes = await image.read()
    label = classify_image(img_bytes)
    return JSONResponse(content={"category": label})

@app.post("/recommend")
async def recommend_outfit(
    occasion: Annotated[str, Form()],
    pinterest_occasions: Annotated[List[str], Form()],
    pinterest_images: List[UploadFile] = File(...),
    wardrobe_images: List[UploadFile] = File(...)
):
    threshold = 0.3
    matched_outfits = []

    print(f"Requested occasion: {occasion}")
    print(f"Received Pinterest occasion tags: {pinterest_occasions}")

    for i, p_img in enumerate(pinterest_images):
        if i >= len(pinterest_occasions):
            print(f"Skipping {p_img.filename} — no matching occasion provided")
            continue

        p_img_occasion = pinterest_occasions[i]
        print(f"{p_img.filename} is tagged as {p_img_occasion}")

        if occasion.strip().lower() != p_img_occasion.strip().lower():
            print(f"Skipping {p_img.filename} — doesn't match requested occasion")
            continue

        p_bytes = await p_img.read()
        p_features = extract_features(p_bytes)
        if p_features is None:
            continue

        matches = []
        for w_img in wardrobe_images:
            w_bytes = await w_img.read()
            category = classify_image(w_bytes)
            w_features = extract_features(w_bytes)
            if w_features is None:
                continue

            similarity = cosine_similarity(p_features, w_features)[0][0]
            print(f"{p_img.filename} vs {w_img.filename} — similarity: {similarity:.2f}")

            if similarity >= threshold:
                matches.append({
                    "wardrobe_image": w_img.filename,
                    "similarity": float(similarity),
                    "category": category
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
