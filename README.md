# AI Generated Care Plan System

## Overview

AI Generated Care Plan System is a healthcare application developed using FastAPI and Machine Learning. The system predicts multiple diseases and automatically generates personalized care plans based on patient health parameters.

## Features

* Diabetes Prediction
* Heart Disease Prediction
* Stroke Risk Prediction
* Kidney Disease Prediction
* Liver Disease Prediction
* BMI Calculation
* Personalized Care Plan Generation
* Risk Level Assessment
* Diet Recommendations
* Exercise Recommendations
* Medication Suggestions
* Monitoring Guidelines
* Follow-up Recommendations

## Technologies Used

* Python
* FastAPI
* Pandas
* Scikit-learn
* Logistic Regression
* Uvicorn

## Datasets Used

* Diabetes Dataset
* Heart Disease Dataset
* Stroke Dataset
* Kidney Disease Dataset
* Liver Disease Dataset

## Model Performance

### Diabetes Model

* Accuracy: 76.62%
* Precision: 69.39%
* Recall: 61.82%
* F1 Score: 65.38%

### Heart Disease Model

* Accuracy: 75.00%
* Precision: 100.00%
* Recall: 66.67%
* F1 Score: 80.00%

### Stroke Model

* Accuracy: 90.00%
* Precision: 85.29%
* Recall: 96.67%
* F1 Score: 90.62%

### Kidney Disease Model

* Accuracy: 83.33%
* Precision: 85.45%
* Recall: 95.92%
* F1 Score: 90.38%

### Liver Disease Model

* Accuracy: 85.71%
* Precision: 87.50%
* Recall: 96.55%
* F1 Score: 91.80%

## Running the Project

```bash
uvicorn app.main:app --reload
```

## API Endpoint

POST /generate-care-plan

The API accepts patient health information and returns:

* Disease Predictions
* Risk Level
* Personalized Care Plan
* Follow-up Recommendations
