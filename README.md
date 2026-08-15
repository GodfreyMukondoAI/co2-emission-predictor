# CO₂ Emission Predictor

A machine-learning application that predicts vehicle CO₂ emissions using:

- Engine Size
- Combined Fuel Consumption

The project uses Multiple Linear Regression with StandardScaler
for feature preprocessing.

## Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Feature Selection

↓

Train/Test Split

↓

StandardScaler

↓

Multiple Linear Regression

↓

Model Evaluation

↓

Saved Model

↓

FastAPI

↓

CO₂ Prediction

## Features

The model uses two input features:

1. `ENGINESIZE`
2. `FUELCONSUMPTION_COMB_MPG`

The target variable is:

`CO2EMISSIONS`

The prediction is returned in:

`g/km`

## Model Evaluation

The model is evaluated using:

- R² Score
- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error

## Project Structure

```text
co2-emission-predictor/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── schemas/
│   │   └── prediction.py
│   ├── services/
│   │   └── prediction_service.py
│   └── ml/
│       ├── model.py
│       └── preprocessing.py
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore