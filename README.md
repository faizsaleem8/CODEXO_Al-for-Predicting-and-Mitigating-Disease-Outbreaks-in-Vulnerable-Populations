# CODEXO_Al-for-Predicting-and-Mitigating-Disease-Outbreaks-in-Vulnerable-Populations

## Overview

This Flask application predicts disease outbreak risks in specific regions using a trained machine learning model. It provides actionable insights such as the outbreak risk (high or low), the confidence level of the prediction, current weather data, and details about nearby healthcare resources. Additionally, the interface features dynamic state and district selection, along with real-time weather integration through the OpenWeatherMap API.

## Features
Dynamic Dropdown: Users can select states and districts dynamically based on data availability.

Weather Data Integration: Fetches real-time weather data (temperature, humidity, rainfall) for selected districts using the OpenWeatherMap API.

Prediction Engine: Predicts the risk of a disease outbreak using key factors like climate, population, and healthcare statistics.

Interactive Visualization: Provides confidence levels for predictions and additional details such as symptoms and prevention strategies.

Scalability: Designed to accommodate additional diseases, regions, and datasets.

## Tech Stack

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript (with AJAX for dynamic elements)

Machine Learning: RandomForestClassifier (scikit-learn)

Data Visualization: Matplotlib for feature importance visualization

API Integration: OpenWeatherMap API

Data Sources: CSV files containing climate, population, and healthcare statistics.

## Installation and Setup.
### Clone the repository:

 
bash


    git clone https://github.com/yourusername/disease-outbreak-prediction.git
    cd disease-outbreak-prediction


## Install required dependencies:

### Prepare your data:


Place the following CSV files in the root directory:

* climate_data_monthly.csv

* population_data.csv

* healthcare_data.csv

* Ensure your trained model (disease_outbreak_model.pkl) and  scaler (scaler.pkl) are also in the root directory.

### Add your OpenWeatherMap API key:

Replace API_KEY in app.py with your actual API key.
Run the application:

bash

    python app.py

** Access the app: Open your browser and navigate to http://127.0.0.1:5000 **

bash 

    pip install pandas scikit-learn joblist matplotlib

### Model Training
The model is trained using a RandomForestClassifier to predict disease outbreak risks based on key features. Below are the steps used for model training:

### Data Preparation:

Merges datasets containing climate, population, and healthcare data.

Extracts features like temperature, humidity, rainfall, population density, and healthcare availability.
