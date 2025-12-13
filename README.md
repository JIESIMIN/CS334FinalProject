# CS334 Final Project
Diabetes status prediction and risk factor analysis.

- Preprocessing (Catherine): standardize numerical values (StandardScalar), variance filtering (remove features with near 0 variance)
- Feature selection (Catherine): Pearson correlation

- Logistic regression (Catherine): LASSO/Ridge (for feature selection)
- KNN (Nicky): Tuned the number of neighbors (k), distance metric, and voting weights to optimize multiclass classification performance.
- Random Forest (Nicky): Tuned key hyperparameters including the number of trees, maximum tree depth, minimum samples per leaf, and minimum samples required for node splitting.
- Gradient boost (Nicky): Tuned the learning rate, number of estimators, and maximum tree depth to improve predictive performance and generalization
- XGboost (Jasmine):
- FNN (Jasmine): 

For each model: 5-fold CV, GridSearchCV (hyperparameter selection), SHAP analysis (top 10)
Metrics: Acc, Precision, Recall, F1, AUCROC

Holdout test set: 10% (25,368 samples)
Training set: 90% 

# Run KNN, Random Forest, Gradient Boost
Open and run: knn_RF_GB.ipynb

Dependencies (imports used in this notebook):
Core: numpy, pandas
Modeling (scikit-learn): train_test_split, GridSearchCV, Pipeline, StandardScaler, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier
Metrics (scikit-learn): accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
Explainability: shap
Visualization: matplotlib, seaborn

We set RANDOM_STATE = 42 for reproducibility.
