from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import uvicorn

# Load the pre-trained model
model = load_model(
    "modeldentifycare.keras"
)  # Ensure 'modeldentifycare.keras' is in the same directory or provide correct path

class_name = [
    "calculus",
    "caries",
    "gingivitis",
    "hypodontia",
    "mouth ulcer",
    "tooth discoloration",
]

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "DENTIFYCARE"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(file.file)

        # Preprocess the image
        image = image.resize((224, 224))  # InceptionV3 input size
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Make prediction
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_name[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        return {"prediction": predicted_class, 
                "confidence": f"{prediction[0][prediction.argmax()] * 100:.2f} %"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
