# CS334 Final Project
Diabetes status prediction and risk factor analysis.

- Preprocessing (Catherine): standardize numerical values (StandardScalar), variance filtering (remove features with near 0 variance)
- Feature selection (Catherine): Pearson correlation

- Logistic regression (Catherine): LASSO/Ridge (for feature selection)
- KNN (Nikki): tune K
- Random Forest (Nikki): tune max tree depth, etc.
- Gradient boost (Nikki): 
- XGboost (Jasmine):
- FNN (Jasmine): 

For each model: 5-fold CV, GridSearchCV (hyperparameter selection), SHAP analysis (top 10)
Metrics: Acc, Precision, Recall, F1, AUCROC

Holdout test set: 10% (25,368 samples)
Training set: 90% 

