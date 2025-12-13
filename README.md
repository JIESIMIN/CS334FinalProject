# Diabetes Status Prediction and Risk Factor Analysis
This project focuses on predicting diabetes status and identifying key risk factors using a combination of classical machine learning models and neural networks.

# Project Overview
## Goal: 
- Predict diabetes status
- Analyze and interpret important risk factors driving predictions
## Approach
- Feature preprocessing and selection
- Multiple supervised ML learning models
- Hyperparameter tuning with cross-validation
- Model explainability using SHAP

# Data Splits
- Training set: 90% of data
- Holdout test set: 10% (25,368 samples)
- Cross-validation: 5-fold CV applied on training data

# Preprocessing & Feature Engineering
- Numerical Standardization
- Variance filtering
- Feature selection

# Models Implemented
We implemented Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, XGBoost, and Feedforward Neural Network.

# Model Training & Evaluation 
## For each model: 
- 5-fold CV
- Hyperparameter tuning via GridSearchCV
- SHAP analysis to identify the top 5 most important features
## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- AUROC

# Reproducibility
RANDOM_STATE = 42 used consistently across experiments

# How to Run the Code
## Core dependencies
numpy, pandas, train_test_split, GridSearchCV, Pipeline, StandardScaler, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer, shap, matplotlib, seaborn
## Model-specific dependencies
KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, xgboost

## KNN, Random Forest, Gradient Boosting
Open and run: knn_RF_GB.ipynb

## XGBoost
Open and run: XGBoost.ipynb

## FNN
Open and run: FNN.ipynb
