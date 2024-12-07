from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from google.cloud import storage
import os

# GCS bucket and model configuration
BUCKET_NAME = "dentifycare_model"  # Replace with your GCS bucket name
MODEL_FILE_NAME = "modeldentifycare.keras"  # Replace with the name of your model file in GCS
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILE_NAME}"  # Temporary directory in Cloud Run or local machine

# Class names for the tooth disease categories
class_name = [
    "calculus",
    "caries",
    "gingivitis",
    "hypodontia",
    "mouth ulcer",
    "tooth discoloration",
]

app = FastAPI()

# Global variable to hold the model
model = None

# Function to download the model from GCS to the local temporary directory
def download_model_from_gcs():
    """
    Downloads the model file from Google Cloud Storage to the local file system.
    """
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading model from GCS bucket: {BUCKET_NAME}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the model during FastAPI application startup.
    """
    global model
    download_model_from_gcs()  # Download model from GCS
    model = load_model(LOCAL_MODEL_PATH)  # Load the model from the local file system
    print("Model loaded successfully.")

@app.get("/")
async def root():
    return {"message": "DENTIFYCARE"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict tooth disease from an uploaded image.
    """
    try:
        # Read and preprocess the image
        image = Image.open(file.file)
        image = image.resize((224, 224)) 
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)  # Inception V3 preprocess input

        # Perform prediction
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)  # Get the class index with the highest confidence
        predicted_class = class_name[predicted_class_index]  # Get the predicted class name
        confidence = prediction[0][predicted_class_index]  # Get the confidence value

        return {
            "Diagnosis": predicted_class,
            "Accuracy": f"{confidence * 100:.2f} %"  # Return confidence as percentage
        }

    except Exception as e:
        return {"error": str(e)}