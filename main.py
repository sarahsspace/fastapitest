from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import os
import nest_asyncio
import uvicorn
from typing import List

nest_asyncio.apply()
app = FastAPI()

model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")

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

@app.post("/recommend")
async def recommend_outfit(
    pinterest_images: List[UploadFile] = File(...),
    pinterest_occasions: List[str] = Form(...),
    wardrobe_images: List[UploadFile] = File(...),
    occasion: str = Form(...)
):
    threshold = 0.3
    matched_outfits = []

    print(f"Request occasion: {occasion}")
    print(f"Received Pinterest occasion tags: {pinterest_occasions}")

    for i, p_img in enumerate(pinterest_images):
        if i >= len(pinterest_occasions):
            print(f"Skipping {p_img.filename} — no occasion provided.")
            continue

        p_img_occasion = pinterest_occasions[i]
        print(f"{p_img.filename} tagged as: {p_img_occasion}")

        if occasion.strip().lower() != p_img_occasion.strip().lower():
            print(f"Skipping {p_img.filename} — doesn't match requested occasion.")
            continue

        p_bytes = await p_img.read()
        p_features = extract_features(p_bytes)
        if p_features is None:
            continue

        matches = []
        for w_img in wardrobe_images:
            w_bytes = await w_img.read()
            w_features = extract_features(w_bytes)
            if w_features is None:
                continue

            similarity = cosine_similarity(p_features, w_features)[0][0]
            print(f"{p_img.filename} vs {w_img.filename} — similarity: {similarity:.2f}")

            if similarity >= threshold:
                matches.append({
                    "wardrobe_image": w_img.filename,
                    "similarity": float(similarity)
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

# Render will run this section (and locally if needed)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
