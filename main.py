from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import joblib
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os

app = FastAPI()

# Load your trained model safely
try:
    model = joblib.load("best_paddy_rf_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # resize to model input size
    image = np.array(image) / 255.0   # normalize
    image = image.flatten().reshape(1, -1)  # flatten for ML models
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Model not loaded."}
        )
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No file uploaded or file is empty."}
            )
        
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Uploaded file is not a valid image."}
            )
        
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        return {"prediction": str(prediction)}
    
    except Exception as e:
        # Catch-all error
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 3000))  # default to 3000 for local testing
    uvicorn.run(app, host="0.0.0.0", port=port)
