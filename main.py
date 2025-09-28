import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import joblib
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
import io

app = FastAPI()


# --- feature extraction (from your training code) ---
def extract_features_from_pil(image_pil, IMG_SIZE=(128,128), hist_bins=32, glcm_levels=8, hog_orient=6, hog_ppc=(16,16)):
    # Convert PIL to RGB numpy
    img = np.array(image_pil.convert("RGB"))
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # HSV stats
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0,1))
    h_std, s_std, v_std = np.std(hsv, axis=(0,1))

    # Color histogram
    hist_r = cv2.calcHist([img_resized],[0],None,[hist_bins],[0,256]).flatten()
    hist_g = cv2.calcHist([img_resized],[1],None,[hist_bins],[0,256]).flatten()
    hist_b = cv2.calcHist([img_resized],[2],None,[hist_bins],[0,256]).flatten()
    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist / (hist.sum() + 1e-8)

    # GLCM features
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    q = (gray * (glcm_levels / 256.0)).astype('uint8')
    glcm = graycomatrix(q, distances=[1], angles=[0], levels=glcm_levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    glcm_feats = np.array([contrast, correlation, energy, homogeneity])

    # HOG features
    hog_feats = hog(gray, orientations=hog_orient, pixels_per_cell=hog_ppc,
                    cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

    # Concatenate all
    feats = np.concatenate([
        np.array([h_mean, s_mean, v_mean, h_std, s_std, v_std]),
        glcm_feats,
        hist,
        hog_feats
    ])
    return feats.reshape(1, -1)

# --- Health endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running!"}

# --- Predict endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load model safely
    try:
        model_path = os.path.join(os.path.dirname(__file__), "best_paddy_rf_model.pkl")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)
        model = None

    if model is None:
        return JSONResponse(status_code=500, content={"status":"error","message":"Model not loaded"})
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"status":"error","message":"Empty file"})
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"status":"error","message":"Invalid image"})

        # Extract features like training
        features = extract_features_from_pil(image)
        prediction = model.predict(features)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message": str(e)})

# --- Run server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
