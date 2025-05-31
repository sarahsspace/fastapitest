from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Use lightweight MobileNetV2 for feature extraction
model = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", pooling='avg'
)

def extract_features(image_bytes: bytes) -> np.ndarray:
    try:
        print("Extracting features...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        features = model.predict(img_array)
        print(f"Feature vector shape: {features.shape}")
        return features[0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/recommend")
async def recommend_outfit(
    pinterest_images: List[UploadFile] = File(...),
    wardrobe_images: List[UploadFile] = File(...),
    wardrobe_categories: str = Form(...),
    pinterest_occasions: str = Form(...)
):
    try:
        print("Pinterest images received:", len(pinterest_images))
        print("Wardrobe images received:", len(wardrobe_images))

        pin_occ = json.loads(pinterest_occasions)
        ward_cats = json.loads(wardrobe_categories)

        pin_feats = []
        for img in pinterest_images:
            bytes_data = await img.read()
            feat = extract_features(bytes_data)
            if feat is not None:
                pin_feats.append(feat)

        ward_feats = []
        for img, cat in zip(wardrobe_images, ward_cats):
            bytes_data = await img.read()
            feat = extract_features(bytes_data)
            if feat is not None:
                ward_feats.append((feat, cat))

        if not ward_feats:
            return JSONResponse(content={"error": "No valid wardrobe features extracted."}, status_code=400)

        recommendations = []
        for i, pin_feat in enumerate(pin_feats):
            matched = {}
            for feat, cat in ward_feats:
                score = cosine_similarity([pin_feat], [feat])[0][0]
                print(f"Cosine similarity with category {cat}: {score:.4f}")
                if cat not in matched or matched[cat]['score'] < score:
                    matched[cat] = {"score": float(score), "category": cat}
            print(f"Matched for Pinterest image {i}: {matched}")
            recommendations.append(matched)

        return {"occasion": pin_occ[0], "matched_outfits": recommendations}

    except Exception as e:
        print("Exception in /recommend:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
