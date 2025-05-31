from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import io
import json
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model for feature extraction
model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", pooling='avg')


def extract_features(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        features = model.predict(img_array, verbose=0)
        return features[0]
    except Exception as e:
        print(f" Error processing image: {e}")
        return None


@app.post("/recommend")
async def recommend_outfit(
    pinterest_images: List[UploadFile] = File(...),
    pinterest_occasions: str = Form(...),
    wardrobe_images: List[UploadFile] = File(...),
    wardrobe_categories: str = Form(...)
):
    try:
        pin_feats = []
        pin_occ = json.loads(pinterest_occasions)

        print(" Pinterest images received:", len(pinterest_images))

        for img in pinterest_images:
            content = await img.read()
            feat = extract_features(content)
            if feat is not None:
                pin_feats.append(feat)

        if not pin_feats:
            return JSONResponse(content={"error": "No valid Pinterest features extracted."}, status_code=400)

        print(" Wardrobe images received:", len(wardrobe_images))
        ward_feats = []
        ward_cats = json.loads(wardrobe_categories)
        for i, img in enumerate(wardrobe_images):
            content = await img.read()
            feat = extract_features(content)
            if feat is not None:
                ward_feats.append((feat, ward_cats[i]))

        if not ward_feats:
            return JSONResponse(content={"error": "No valid wardrobe features extracted."}, status_code=400)

        recommendations = []
        for i, pin_feat in enumerate(pin_feats):
            matched = {}
            for feat, cat in ward_feats:
                score = cosine_similarity([pin_feat], [feat])[0][0]
                print(f" Sim: {score:.3f} ➜ Pin {i} ↔ {cat}")
                if cat not in matched or matched[cat]['score'] < score:
                    matched[cat] = {"score": float(score), "category": cat}

            recommendations.append(matched)

        return {"occasion": pin_occ[0], "matched_outfits": recommendations}

    except Exception as e:
        print(" Exception in /recommend:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)