# Explainable AI for Lung Image Classification

## Overview
This project explores the application of deep learning and Explainable AI (XAI) techniques to classify lung images and interpret model predictions. The model is trained to distinguish between different lung conditions (e.g., Normal vs Abnormal), and multiple XAI methods are used to visualize how the model makes decisions.

## Objective
The goal of this project is to:
- Build a deep learning model for lung image classification
- Apply Explainable AI techniques to interpret predictions
- Understand which regions of the image influence model decisions

## Tools & Technologies
- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Google Colab
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- LIME (Local Interpretable Model-Agnostic Explanations)
- SHAP (SHapley Additive exPlanations)

## Methodology

### 1. Data Preparation
- Loaded lung image dataset from Google Drive
- Resized images to 224×224
- Created training and validation datasets

### 2. Model Development
- Used MobileNetV2 as a pre-trained base model
- Added custom classification layers
- Trained the model using transfer learning

### 3. Prediction
- Generated predictions on validation images
- Classified images into respective categories

### 4. Explainable AI Techniques

#### Grad-CAM
- Highlights important regions in the image
- Shows where the model is focusing during prediction

#### LIME
- Explains individual predictions locally
- Identifies important superpixels influencing the output

#### SHAP
- Measures contribution of features to predictions
- Provides global and local interpretability

## Key Work Completed
- Implemented a deep learning classification pipeline
- Applied multiple XAI techniques (Grad-CAM, LIME, SHAP)
- Visualized model decision-making process
- Analyzed interpretability of predictions

## Results
The model successfully classifies lung images and provides visual explanations for its predictions. XAI methods helped identify key regions influencing model decisions, improving transparency and understanding of the model.

## Repository Contents
- `lung_image_xai.ipynb` — Complete notebook with training and XAI implementation
- `README.md` — Project documentation

## Future Improvements
- Improve model accuracy with larger dataset
- Apply advanced architectures (e.g., EfficientNet)
- Extend to multi-class classification
- Optimize XAI visualizations

## Author
Kaniz Fatema Ankan
