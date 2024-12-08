# Dentifycare-ML
Dental Disease Classification with Pretrained InceptionV3 Model
This project involves fine-tuning a pretrained InceptionV3 model to classify dental diseases into six categories. The model is trained on a custom dataset of dental images using TensorFlow and Keras.

The model employs data augmentation, transfer learning, and fine-tuning techniques to achieve high accuracy on validation and test datasets.

Project Overview
Dataset:

Training, validation, and test datasets are organized in respective directories.
Classes: 6 categories of dental diseases.
Images are resized to (224, 224) for input into the model.
Model Architecture:

Base model: InceptionV3 (pretrained on ImageNet).
Top layers: Flattening, fully connected layers, dropout, and softmax for classification.
Data augmentation is applied to improve generalization.
Training Details:

Loss function: sparse_categorical_crossentropy.
Optimizer: Adam with learning rate scheduling.
Metrics: Accuracy.
Early stopping and learning rate reduction callbacks are used.
Performance:

Validation Accuracy: ~92.78%.
Test Accuracy: ~92.70%.
File Structure
bash
Copy code
project/
├── modeldentifycare.keras          # Saved trained model
├── train/                          # Training dataset
├── val/                            # Validation dataset
├── test/                           # Test dataset
└── README.md                       # This README file
Setup and Requirements
Prerequisites
Python 3.8 or higher
TensorFlow 2.0 or higher
Required Python packages:
tensorflow
matplotlib
google.colab (for Google Colab usage)
Installation
Clone the repository:
bash
Copy code
git clone <repository-url>
Install required dependencies:
bash
Copy code
pip install tensorflow matplotlib
Model Training and Evaluation
Step 1: Mount Google Drive
Ensure your dataset is organized in the following structure:

bash
Copy code
/content/drive/MyDrive/capstone-jun/split_dataset/
    ├── train/
    ├── val/
    └── test/
Mount the drive:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Step 2: Preprocess Dataset
Define the dataset preprocessing pipeline:

python
Copy code
from tensorflow.keras.utils import image_dataset_from_directory
training_dataset, validation_dataset, test_dataset = dataset()
Step 3: Train the Model
Train the base or fine-tuned model using:

python
Copy code
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
)
Step 4: Evaluate the Model
Evaluate test accuracy and loss:

python
Copy code
test_loss, test_accuracy = model.evaluate(test_dataset)
Step 5: Save and Load Model
Save the trained model:

python
Copy code
model.save('modeldentifycare.keras')
Load the model for inference:

python
Copy code
from tensorflow.keras.models import load_model
model = load_model('modeldentifycare.keras')
Pretrained model link: Google Drive

Results
Training Accuracy: ~96.9%
Validation Accuracy: ~92.78%
Test Accuracy: ~92.70%
Accuracy and Loss Graph
<img src="accuracy_loss_plot.png" alt="Accuracy and Loss Graph" width="600">
References
TensorFlow Documentation
InceptionV3 Paper
Pretrained model weights: ImageNet.
