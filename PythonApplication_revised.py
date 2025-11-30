
import argparse 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score,precision_recall_curve, auc, roc_curve
import time
import shap


def standard_scale(xTrain, xTest): 
    # initialize scaler to scale data to have a mean of 0 and SD of 1 
    scaler = StandardScaler() 
    # standardize numerical variables 

    # fit scaler on training data
    scaler.fit(xTrain)

    xtrain_scaled = scaler.transform(xTrain)
    xtest_scaled = scaler.transform(xTest)

    return xtrain_scaled, xtest_scaled

def cal_corr(df): 
    # calculates the pearson correlation for numeric features 
    corrDF = df.corr() 
    return corrDF 


def select_features(trainDF, testDF): 
    corr_matrix = cal_corr(trainDF) 

    clms = corr_matrix.columns
    #check the correlation with the target to be smaller than 1%, then we drop the column
    gamma = 0.015        #threshold for correlation with target
    #find out highly correlated features
    sigma = 0.80         #threshold for two highly correlated features 

    last_row_ind = clms.size-1
    columnsToDrop = []
    #Find highly correlated columns, drop the one that has less correlation with the target
    for col in range(clms.size-1):
        for row in range(col+1,clms.size-1):       #all rows below the element excep the last row
            if(abs(corr_matrix.iloc[row, col]) > sigma):
                # compare correlation with target (last row)
                if(abs(corr_matrix.iloc[last_row_ind, col]) > abs(corr_matrix.iloc[last_row_ind, row])):
                    columnsToDrop.append(clms[row])
                else:
                    columnsToDrop.append(clms[col])
    # drop features with very low correlation with target 
    worstIndex = 0
    worstCorr = corr_matrix.iloc[last_row_ind,0]
    for index, item in enumerate(clms):
            # low correlation with target
            if(abs(corr_matrix.iloc[last_row_ind,index]) < gamma and (item not in columnsToDrop)):
                columnsToDrop.append(item)
            # track overall worst feature
            if(abs(corr_matrix.iloc[last_row_ind,index]) < abs(worstCorr)):
                worstIndex = index
                worstCorr = corr_matrix.iloc[last_row_ind,worstIndex]

    # Need to drop the worst feature in any case
    if len(columnsToDrop) < 1:
        columnsToDrop.append(clms[worstIndex])
    # drop columns from both train and test 
    trainDfSel = trainDF.drop(columns=columnsToDrop)
    testDfSel = testDF.drop(columns=columnsToDrop)

    return trainDfSel, testDfSel

def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    start = time.time()
    #param_grid = {'alpha': alphaList} 

    grid_search = GridSearchCV(estimator=clf, param_grid=pgrid, cv=5)
    # 'cv' specifies the number of cross-validation folds
    # 'scoring' defines the metric to optimize (e.g., 'neg_mean_squared_error' for minimizing MSE)

    grid_search.fit(xTrain, yTrain)
    timeElapsed = time.time() - start

    best_model = grid_search.best_estimator_

    myReturn1 = {}
    # Get probability predictions for the positive class on the training data
    # DecisionTreeClassifier.predict_proba() returns probabilities for all classes.
    # We need the probability of the positive class (usually the second column for binary classification).
    myReturn1["Time"] = timeElapsed

    yTestPred = best_model.predict(xTest)
    # Calculate F1 score for binary classification
    testF1 = f1_score(yTest, yTestPred, average='weighted')
    myReturn1["F1"] = testF1

    myReturn2 = grid_search.best_params_

    return myReturn1, myReturn2, best_model

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    start = time.time()

    # Initialize RandomizedSearchCV
    # n_iter: Number of parameter settings that are sampled.
    # cv: Number of folds for cross-validation.
    # scoring: Metric to evaluate the model performance.
    # n_jobs: Number of jobs to run in parallel (-1 means use all available cores).
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=pgrid,
        n_iter=1,            # Try 1 different random combinations of hyperparameters
        cv=5
    )

    # Fit the RandomizedSearchCV object to the data
    random_search.fit(xTrain, yTrain)

    # Print the best parameters and best score found
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Cross-Validation Score: {random_search.best_score_}")

    # Access the best estimator found
    best_model = random_search.best_estimator_
    
    timeElapsed = time.time() - start

    myReturn1 = {}
    # Get probability predictions for the positive class on the training data
    # DecisionTreeClassifier.predict_proba() returns probabilities for all classes.
    # We need the probability of the positive class (usually the second column for binary classification).
    myReturn1["Time"] = timeElapsed

    yTestPred = best_model.predict(xTest)
    # Calculate F1 score for binary classification. F1=1 is perfect accuracy, 0 is worst accuracy
    testF1 = f1_score(yTest, yTestPred, average='weighted')
    myReturn1["F1"] = testF1

    myReturn2 = random_search.best_params_

    return myReturn1, myReturn2

def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, bestParamDict):

    cls_perf, gs_p, gs_model = eval_gridsearch(clf, clfGrid, xTrain, yTrain, xTest, yTest)
    perfDict[clfName + " (Grid)"] = cls_perf

    clfr_perf, rs_p  = eval_randomsearch(clf, clfGrid, xTrain, yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf

    bestParamDict[clfName] = {
    "Grid": gs_p,
    "GridModel": gs_model,
    "Random": rs_p
    }
    return perfDict, bestParamDict

def get_parameter_grid(mName):
    if mName == "LR (L1)":
        return {
            'C': [0.01, 0.1, 1.0],
            'max_iter': [1000, 2000],
            # solver and multi_class fixed → do NOT search
        }

    if mName == "LR (L2)":
        return {
            'C': [0.01, 0.1, 1.0],
            'max_iter': [1000, 2000],
        }

    return {}


def main(): 
    # read csv 
    df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
    # split into features and target 
    y = df['Diabetes_012']                # Target
    x = df.drop('Diabetes_012', axis=1)   # Features

    # split data in to train and test sets 
    xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, train_size=0.9, test_size=0.1, random_state=42)

    # standardize features 
    x_train_scaled, x_test_scaled = standard_scale(xTrain, xTest)

    # convert scaled features back to dataframes
    trainDF = pd.DataFrame(x_train_scaled, columns=xTrain.columns)
    testDF = pd.DataFrame(x_test_scaled, columns=xTest.columns)

    # add target column back as last column 
    trainDF['Diabetes_012'] = yTrain.values
    testDF['Diabetes_012'] = yTest.values 

    # feature selection using correlation 
    trainDF_sel, testDF_sel = select_features(trainDF, testDF)

    # separate features and target again from the *selected* data
    y_train_sel = trainDF_sel['Diabetes_012']
    X_train_sel = trainDF_sel.drop('Diabetes_012', axis=1)

    y_test_sel = testDF_sel['Diabetes_012']
    X_test_sel = testDF_sel.drop('Diabetes_012', axis=1)

    X_train_sel.to_csv("x_train.csv")
    y_train_sel.to_csv("y_train.csv")
    X_test_sel.to_csv("x_test.csv")
    y_test_sel.to_csv("y_test.csv")



    perfDict = {}
    bestParamDict = {}

    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)

    lassoClf = LogisticRegression(
    penalty='l1',
    solver='saga',
    multi_class='multinomial',
    max_iter=2000
)
    perfDict, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   X_train_sel, y_train_sel, X_test_sel, y_test_sel,
                                                   perfDict, bestParamDict)
    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
 
    ridgeClf = LogisticRegression(
    penalty='l2',
    solver='saga',
    multi_class='multinomial',
    max_iter=2000
)
    perfDict, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   X_train_sel, y_train_sel, X_test_sel, y_test_sel,
                                                   perfDict, bestParamDict)
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')

    print("Running SHAP analysis...")

    best_l1 = bestParamDict["LR (L1)"]["GridModel"]

    # Use the selected training data as background and evaluation data
    explainer = shap.LinearExplainer(best_l1, X_train_sel)
    shap_values = explainer.shap_values(X_train_sel)

    # Inspect shape once (optional, for debugging)
    print("TYPE:", type(shap_values))
    print("SHAP output shape:", np.array(shap_values).shape)

    # ----- Handle different SHAP output formats -----
    class_idx = 2  # Diabetes_012 has classes 0,1,2 → we pick class 2

    if isinstance(shap_values, list):
        # Classic multiclass: list of arrays [class0, class1, class2]
        shap_values_class = shap_values[class_idx]          # (n_samples, n_features)
    else:
        # 3D array: (n_samples, n_features, n_classes)
        # or: (n_samples, n_classes, n_features) in some versions
        arr = np.array(shap_values)
        if arr.ndim == 3:
            # Try to detect which axis is "class"
            # Case A: (samples, features, classes)
            if arr.shape[2] <= 5:   # small number of classes
                shap_values_class = arr[..., class_idx]     # (n_samples, n_features)
            # Case B: (samples, classes, features)
            elif arr.shape[1] <= 5:
                shap_values_class = arr[:, class_idx, :]    # (n_samples, n_features)
            else:
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
        elif arr.ndim == 2:
            # Binary case: already (n_samples, n_features)
            shap_values_class = arr
        else:
            raise ValueError(f"Unexpected SHAP ndim: {arr.ndim}")

    # Sanity check
    print("Final SHAP class matrix shape:", shap_values_class.shape)
    print("X_train_sel shape:", X_train_sel.shape)

    # =====================
    # 1️⃣ BAR PLOT (mean |SHAP|)
    # =====================
    shap.summary_plot(
        shap_values_class,
        X_train_sel,                      # use same DataFrame
        feature_names=X_train_sel.columns,
        plot_type="bar",
        max_display=20
    )

    # =====================
    # 2️⃣ BEESWARM PLOT
    # =====================
    shap.summary_plot(
        shap_values_class,
        X_train_sel,
        feature_names=X_train_sel.columns,
        max_display=20
    )


    


    print(perfDF)

if __name__ == "__main__":

    main()
