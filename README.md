# Disease Prediction System

A comprehensive Streamlit web application for predicting multiple diseases using machine learning models.

## Features

- **Four Disease Models**: Diabetes, Heart Disease, Kidney Disease, and Thyroid Disease
- **Multiple ML Algorithms**: KNN, Naive Bayes, Logistic Regression, and Decision Tree
- **Evaluation Metrics**: View accuracy, precision, recall, F1 score, and confusion matrices for each model
- **Interactive Predictions**: Input patient data and get predictions from all models
- **Input Range Suggestions**: Helpful tooltips showing valid input ranges for each feature

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Models

First, train all the disease models by running:

```bash
python diabetes_model.py
python heart_disease_model.py
python kidney_disease_model.py
python thyroid_disease_model.py
```

This will create pickle files (`*_models.pkl`) containing the trained models and preprocessing objects.

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Application Structure

### Tabs

1. **Diabetes Tab**: Predict diabetes based on:
   - Pregnancies, Glucose, Blood Pressure, Skin Thickness
   - Insulin, BMI, Diabetes Pedigree Function, Age

2. **Heart Disease Tab**: Predict heart disease based on:
   - Age, Gender, Blood Pressure, Cholesterol
   - Exercise habits, Smoking, Family history
   - BMI, Various health indicators

3. **Kidney Disease Tab**: Predict kidney disease based on:
   - Blood pressure, Specific gravity, Albumin
   - Blood urea, Serum creatinine, Electrolytes
   - Blood cell counts, Hypertension status

4. **Thyroid Disease Tab**: Predict thyroid disease recurrence based on:
   - Age, Gender, Smoking history
   - Thyroid function, Physical examination
   - Pathology, Stage, Response to treatment

### Features

- **Evaluation Metrics**: Each tab displays comprehensive evaluation metrics for all four models
- **Confusion Matrices**: Visual representation of model performance
- **Prediction Interface**: User-friendly input forms with range suggestions
- **Multi-Model Predictions**: Get predictions from all models simultaneously
- **Confidence Scores**: View prediction confidence percentages

## File Structure

```
.
├── app.py                      # Main Streamlit application
├── diabetes_model.py           # Diabetes model training script
├── heart_disease_model.py      # Heart disease model training script
├── kidney_disease_model.py     # Kidney disease model training script
├── thyroid_disease_model.py    # Thyroid disease model training script
├── requirements.txt            # Python dependencies
├── diabetes.csv                # Diabetes dataset
├── heart_disease.csv           # Heart disease dataset
├── Kidney disease.csv          # Kidney disease dataset
├── Thyroid_Diff.csv            # Thyroid disease dataset
└── README.md                   # This file
```

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Notes

- All models use a train-test split of 80-20
- Models are saved as pickle files for quick loading
- The application caches models for faster performance
- Input validation ensures values are within acceptable ranges

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Scikit-learn 1.3.0+

## Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

