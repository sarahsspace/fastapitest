from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
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

@app.get("/")
def root():
    return {"status": "running"}

# Load EfficientNet for visual style embedding
feature_model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")

def extract_vector(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_model.predict(img_array)
    return features

@app.post("/recommend")
async def recommend_from_pinterest(
    occasion: Annotated[str, Form()],
    pinterest_occasions: Annotated[List[str], Form()],
    wardrobe_categories: Annotated[List[str], Form()],
    pinterest_images: List[UploadFile] = File(...),
    wardrobe_images: List[UploadFile] = File(...)
):
    print("🔁 Received recommend request")

    threshold = 0.3
    results = []

    # Precompute wardrobe feature vectors with category
    wardrobe_items = []
    for i, w_img in enumerate(wardrobe_images):
        if i >= len(wardrobe_categories):
            continue

        category = wardrobe_categories[i].strip().lower()
        if category not in ["shirt", "pants", "shoes", "dress"]:
            continue

        w_bytes = await w_img.read()
        w_vec = extract_vector(w_bytes)

        wardrobe_items.append({
            "filename": w_img.filename,
            "vector": w_vec,
            "category": category
        })

    # For each Pinterest image with matching occasion
    for i, p_img in enumerate(pinterest_images):
        if i >= len(pinterest_occasions):
            continue
        if occasion.strip().lower() != pinterest_occasions[i].strip().lower():
            continue

        p_bytes = await p_img.read()
        p_vec = extract_vector(p_bytes)

        # Track top match per category
        top_matches = {
            "shirt": None,
            "pants": None,
            "shoes": None,
            "dress": None
        }

        # Compare Pinterest image with each wardrobe item
        for item in wardrobe_items:
            sim = cosine_similarity(p_vec, item["vector"])[0][0]
            if sim < threshold:
                continue

            cat = item["category"]
            if top_matches[cat] is None or sim > top_matches[cat]["similarity"]:
                top_matches[cat] = {
                    "filename": item["filename"],
                    "similarity": float(sim)
                }

        # Assemble outfit using rules
        outfit = None
        if top_matches["dress"] and top_matches["shoes"]:
            outfit = {
                "dress": top_matches["dress"],
                "shoes": top_matches["shoes"]
            }
        elif top_matches["shirt"] and top_matches["pants"] and top_matches["shoes"]:
            outfit = {
                "shirt": top_matches["shirt"],
                "pants": top_matches["pants"],
                "shoes": top_matches["shoes"]
            }

        if outfit:
            results.append({
                "pinterest_image": p_img.filename,
                "recommended_outfit": outfit
            })

    return JSONResponse(content={
        "occasion": occasion,
        "matched_outfits": results
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
