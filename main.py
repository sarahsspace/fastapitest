# Import necessary libraries
import nest_asyncio
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import io

# Apply workaround for running async in non-async environments
nest_asyncio.apply()

# Create FastAPI app instance
app = FastAPI()

# Load EfficientNet model once at startup
model = EfficientNetV2B0(weights='imagenet', include_top=False, pooling='avg')

# Feature extraction helper
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

# Main recommendation endpoint
@app.post("/recommend")
async def recommend_outfits(
    pinterest_images: list[UploadFile] = File(...),
    wardrobe_images: list[UploadFile] = File(...),
    occasion: str = Form(...)
):
    try:
        print(f"Received {len(pinterest_images)} Pinterest images and {len(wardrobe_images)} wardrobe images for occasion: {occasion}")

        # Extract features for wardrobe
        wardrobe_features = []
        wardrobe_filenames = []
        for w_img in wardrobe_images:
            content = await w_img.read()
            features = extract_features(content)
            if features is not None:
                wardrobe_features.append(features)
                wardrobe_filenames.append(w_img.filename)

        if not wardrobe_features:
            return JSONResponse(content={"error": "No valid wardrobe features extracted."}, status_code=400)

        # Extract features for Pinterest images
        pinterest_features = []
        pinterest_filenames = []
        for p_img in pinterest_images:
            content = await p_img.read()
            features = extract_features(content)
            if features is not None:
                pinterest_features.append(features)
                pinterest_filenames.append(p_img.filename)

        if not pinterest_features:
            return JSONResponse(content={"error": "No valid Pinterest features extracted."}, status_code=400)

        # Match each Pinterest outfit with the most similar wardrobe items
        matched_outfits = []

        for idx, pinterest_feature in enumerate(pinterest_features):
            similarities = cosine_similarity(pinterest_feature, np.vstack(wardrobe_features))[0]
            # Top 3 wardrobe matches for each Pinterest outfit
            top_indices = np.argsort(similarities)[::-1][:3]
            recommendations = []
            for i in top_indices:
                recommendations.append({
                    "wardrobe_image": wardrobe_filenames[i],
                    "similarity": float(similarities[i])
                })

            matched_outfits.append({
                "pinterest_image": pinterest_filenames[idx],
                "recommended_wardrobe": recommendations
            })

        # Return JSON response
        return JSONResponse(content={
            "occasion": occasion,
            "matched_outfits": matched_outfits
        })

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

# Start server if running locally (Render handles this automatically usually)
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
