# DentifyCare-ML  
**Dental Disease Classification with Pretrained InceptionV3 Model**  

DentifyCare-ML utilizes transfer learning with the InceptionV3 model to classify dental diseases into six categories. This README outlines the dataset, model, tools, training process, and results of the project.  

---

## Dataset  
The dataset used for training and evaluation can be downloaded here:  
[**Dental Disease Dataset**](https://drive.google.com/drive/folders/139hFbLExoB3QEQeYFJRJw_zQt9yjXLE1?usp=sharing)  

## Categories of Dental Diseases:  
1. **Cavities**  
   - **Description**: Damage to the tooth enamel caused by bacterial acid in plaque.  
   - **Symptoms**: Tooth sensitivity, pain, or visible holes.  
   - **Causes**: Poor oral hygiene, high sugar intake.  

2. **Gum Disease**  
   - **Description**: Inflammation or infection of the gums and supporting bone.  
   - **Symptoms**: Red, swollen, bleeding gums; bad breath.  
   - **Causes**: Plaque buildup, smoking, chronic diseases like diabetes.  

3. **Oral Cancer**  
   - **Description**: Cancer in areas such as the tongue, lips, gums, or cheeks.  
   - **Symptoms**: Persistent sores, patches, swelling, or pain.  
   - **Causes**: Smoking, alcohol, UV exposure, HPV.  

4. **Plaque**  
   - **Description**: Sticky layer on teeth from bacteria and food debris.  
   - **Symptoms**: Rough teeth surface, bad breath, dull appearance.  
   - **Causes**: Poor hygiene, infrequent brushing, high sugar diet.  

5. **Tooth Discoloration**  
   - **Description**: Staining of teeth, affecting enamel or dentin.  
   - **Symptoms**: Uneven tooth color, stains, or spots.  
   - **Causes**: Coffee, tea, tobacco, certain medications.  

6. **Healthy**  
   - **Description**: Teeth and gums in optimal health, free of disease.  
   - **Characteristics**: Naturally white teeth, pink gums, no pain or bad breath.  

---

## Dataset Structure  
- **Train**: Labeled images for training.  
- **Validation**: Images for model validation during training.  
- **Test**: Images for evaluating model performance.  

---

## Model Architecture  
The project employs **InceptionV3** (pretrained on ImageNet) with the following modifications:  
- **Flattening Layer**: Converts output into a 1D vector.  
- **Dense Layers**: Added layers with ReLU activation.  
- **Dropout Layers**: Prevent overfitting.  
- **Softmax Layer**: Outputs probabilities for the six categories.  

### Fine-Tuning:  
1. Train the top layers of the model first.  
2. Unfreeze some base model layers and fine-tune the entire model.  

---

## Tools and Libraries  
- **TensorFlow**: Model building, image preprocessing, and training.  
- **NumPy**: Numerical operations and array handling.  
- **Matplotlib**: Visualizing training accuracy and loss.  
- **Google Colab**: Development environment with GPU support.  
- **Kaggle API**: Direct dataset download.  

---

## Training Procedure  
### Dataset Preparation  
- Resize images to (224, 224).  
- Augmentation: `RandomFlip`, `RandomRotation`, and `RandomZoom`.  

### Model Training  
- **Optimizer**: Adam with learning rate scheduling.  
- **Loss Function**: `sparse_categorical_crossentropy`.  
- **Callbacks**:  
  - Early stopping for improved efficiency.  
  - Learning rate scheduler for better convergence.  

---

## Results  
- **Training Accuracy**: ~96.9%  
- **Validation Accuracy**: ~92.78%  
- **Test Accuracy**: ~92.70%  

### Accuracy and Loss Graph  
![Accuracy and Loss Graph](accuracy_loss_plot.png)  

---

## Setup and Usage  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/<username>/DentifyCare-ML.git
   cd DentifyCare-ML
   

2. Install dependencies:
    ```bash
    pip install tensorflow matplotlib numpy kaggle

### Running the Code
1. Preprocess the Dataset
    ```bash
    from tensorflow.keras.utils import image_dataset_from_directory
    training_dataset = image_dataset_from_directory(
    "path_to_training_data",
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
    )

2. Train the Model
    ```bash
    history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
    )

3. Evaluate the Model
    ```bash
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

4. Save the Model
    ```bash
    model.save('model_dentifycare.keras')


## Pretrained Model
Download the pretrained model:
Pretrained Model (https://drive.google.com/file/d/1-39RTzx1TgT9jJWFaKJOUL6rpVQWg8dD/view?usp=drive_link)
