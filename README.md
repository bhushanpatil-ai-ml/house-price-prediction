# House Price Prediction – Advanced Machine Learning Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning regression pipeline for predicting house prices using property-related features.

This project builds a complete machine learning workflow including data preprocessing, exploratory data analysis, outlier handling, log transformation, model comparison, cross-validation, hyperparameter tuning, and final model evaluation.

---

## Table of Contents

- Problem Statement  
- Objective  
- Dataset  
- Technologies Used  
- Machine Learning Models  
- Project Workflow  
- Evaluation Metrics  
- Model Performance  
- Feature Importance  
- Project Structure  
- Results  
- How to Run the Project  
- Future Improvements  
- Author  

---

## Problem Statement

Accurately predicting house prices is a fundamental problem in real estate analytics.

Housing prices depend on multiple factors such as:

- number of bedrooms
- number of bathrooms
- living area
- property grade
- location attributes
- construction year

The challenge is to build a regression model that captures the relationships between these features and accurately predicts property prices.

---

## Objective

The main objectives of this project are:

- Build a complete regression pipeline for house price prediction
- Perform exploratory data analysis and feature analysis
- Detect and remove outliers using IQR
- Apply log transformation to improve target distribution
- Train multiple regression models
- Compare model performance
- Tune hyperparameters using GridSearchCV
- Save the final trained model

---

## Dataset

The dataset used in this project is the **House Sales Dataset**.

Dataset Source  
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

Dataset Characteristics

- Number of houses: 21,613
- Multiple property attributes
- Real estate pricing dataset

Target Variable

price

---

## Technologies Used

### Programming
Python

### Data Science Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Tools

- Git
- GitHub
- Joblib

---

## Machine Learning Models

This project compares multiple regression algorithms:

- Linear Regression (Baseline Model)
- Random Forest Regressor
- XGBoost Regressor

XGBoost generally performs best for tabular datasets.

---

## Project Workflow

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Correlation Analysis  
4. Removing unnecessary columns  
5. Outlier detection and removal using IQR  
6. Log transformation of target variable  
7. Feature and target separation  
8. Train-test split  
9. Model training  
10. Model evaluation  
11. Cross-validation  
12. Hyperparameter tuning using GridSearchCV  
13. Feature importance analysis  
14. Final tuned model saving  

---

## Evaluation Metrics

Regression models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

These metrics help measure prediction accuracy.

---

## Model Performance

| Model | R² Score |
|------|---------|
| Linear Regression | ~0.70 |
| Random Forest | ~0.85 |
| XGBoost | ~0.88 |

After hyperparameter tuning, the **XGBoost model achieved the best performance**.

---

## Feature Importance

The model identifies the most important features influencing house prices.

Top features typically include:

- sqft_living
- grade
- bathrooms
- view
- sqft_above

Feature importance visualization:

outputs/feature_importance.png

---

## Project Structure

house-price-prediction/

├── data/  
│   └── housing.csv  

├── models/  
│   ├── best_house_price_model.pkl  
│   └── tuned_house_price_model.pkl  

├── notebooks/  
│   └── eda.ipynb  

├── outputs/  
│   └── feature_importance.png  

├── src/  
│   ├── data_preprocessing.py  
│   ├── train_model.py  
│   └── evaluate_model.py  

├── .gitignore  
├── README.md  
└── requirements.txt  

---

## Results

The regression pipeline successfully predicts housing prices with high accuracy.

The tuned **XGBoost model outperforms baseline models**, achieving strong predictive performance after feature engineering and hyperparameter tuning.

---

## How to Run the Project

Install dependencies

pip install -r requirements.txt

Train the model

python src/train_model.py

Evaluate the model

python src/evaluate_model.py

---

## Future Improvements

- Add feature scaling
- Implement advanced feature engineering
- Deploy prediction API using FastAPI
- Build interactive UI using Streamlit
- Add experiment tracking using MLflow

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  
Pune, Maharashtra, India