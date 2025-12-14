
import argparse 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import shap
import matplotlib.pyplot as plt 
from sklearn.metrics import (roc_auc_score, 
                             f1_score, 
                             precision_score, 
                             recall_score, 
                             accuracy_score, 
                             precision_recall_curve, 
                             auc, 
                             roc_curve)



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

    grid_search = GridSearchCV(estimator=clf, param_grid=pgrid, cv=5, scoring='roc_auc_ovr')
    # 'cv' specifies the number of cross-validation folds
    # 'scoring' defines the metric to optimize 

    grid_search.fit(xTrain, yTrain)
    timeElapsed = time.time() - start

    best_model = grid_search.best_estimator_

    myReturn1 = {}
    # Get probability predictions for the positive class on the training data
    # DecisionTreeClassifier.predict_proba() returns probabilities for all classes.
    # We need the probability of the positive class (usually the second column for binary classification).
    myReturn1["Time"] = timeElapsed
    y_test_pred_proba = best_model.predict_proba(xTest)

    # Calculate ROC AUC on the training data
    testAUC = roc_auc_score(yTest, y_test_pred_proba, multi_class='ovr', average='macro')
    myReturn1["AUC"] = testAUC

    yTestPred = best_model.predict(xTest)
    print(np.unique(yTest))
    # --- required metrics ---
    # Accuracy
    acc = accuracy_score(yTest, yTestPred)
    myReturn1["Accuracy"] = acc

    # Precision (macro)
    prec_macro = precision_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["Precision_macro"] = prec_macro

    # Recall (macro)
    rec_macro = recall_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["Recall_macro"] = rec_macro

    # F1-score (macro)
    f1_macro = f1_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["F1_macro"] = f1_macro

    best_params = grid_search.best_params_

    ### metrics for each class ###
    precision = precision_score(yTest, yTestPred, labels=[0, 1, 2], average=None, zero_division=0)
    recall = recall_score(yTest, yTestPred, labels=[0, 1, 2], average=None, zero_division=0)
    f1 = f1_score(yTest, yTestPred, labels=[0, 1, 2], average=None, zero_division=0)
    
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("F1 score per class:", f1)
    print("Support:", np.count_nonzero(yTest == 0),np.count_nonzero(yTest == 1),np.count_nonzero(yTest == 2))

    return myReturn1, best_params

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    start = time.time()

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=pgrid,
        n_iter=1,            # Try 1 different random combinations of hyperparameters
        cv=5,
        scoring='roc_auc_ovr'
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
    myReturn1["Time"] = timeElapsed

    y_test_pred_proba = best_model.predict_proba(xTest)
    classes = np.unique(yTest)

    # Multiclass ROC AUC 
    testAUC = roc_auc_score(yTest, y_test_pred_proba, multi_class='ovr', average='macro')
    myReturn1["AUC"] = testAUC

   # ---- Predictions as labels ----
    yTestPred = best_model.predict(xTest)
    print(np.unique(yTest))
    # Accuracy
    acc = accuracy_score(yTest, yTestPred)
    myReturn1["Accuracy"] = acc

    # Precision (macro)
    prec_macro = precision_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["Precision_macro"] = prec_macro

    # Recall (macro)
    rec_macro = recall_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["Recall_macro"] = rec_macro

    # F1-score (macro)
    f1_macro = f1_score(yTest, yTestPred, average='macro', zero_division=0)
    myReturn1["F1_macro"] = f1_macro

    # Best hyperparameters from RandomizedSearchCV
    best_params = random_search.best_params_

    return myReturn1, best_params

def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, bestParamDict):
    # evaluate grid search and add to perfDict
    cls_perf, gs_p  = eval_gridsearch(clf, clfGrid, xTrain,
                                               yTrain, xTest, yTest)
    perfDict[clfName + " (Grid)"] = cls_perf
    # evaluate random search and add to perfDict
    clfr_perf, rs_p  = eval_randomsearch(clf, clfGrid, xTrain,
                                            yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf
    bestParamDict[clfName] = {"Grid": gs_p, "Random": rs_p}
    return perfDict, bestParamDict

def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    searchParam = {}
    if mName == "LR (L1)":                                      #Lasso
        searchParam = {
            'C' : [0.1,1, 10],                                      # Inverse of regularization strength; must be a positive float.
            'solver' : ['saga'],                                # Algorithm to use in the optimization problem.
            'max_iter' : [500]                            # Maximum number of iterations taken for the solvers to converge.
        }
    if mName == "LR (L2)":
        searchParam = {
            'C' : [0.1,1, 10],                                      # Inverse of regularization strength; must be a positive float.
            'solver' : ['saga'],                            # Algorithm to use in the optimization problem.
            'max_iter' : [500]                            # Maximum number of iterations taken for the solvers to converge.
        }

    return searchParam


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
    # fill in
    lassoClf = LogisticRegression(penalty='l1')
    perfDict, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   X_train_sel, y_train_sel, X_test_sel, y_test_sel,
                                                   perfDict, bestParamDict)
    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    # fill in
    ridgeClf = LogisticRegression(penalty='l2')
    perfDict, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   X_train_sel, y_train_sel, X_test_sel, y_test_sel,
                                                   perfDict, bestParamDict)
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')
    print(perfDF)

    corr_with_target = df.corr()['Diabetes_012'].drop('Diabetes_012')
    print(corr_with_target)

    # Sort top positive correlations
    corr_sorted = corr_with_target.sort_values(ascending=False)

    plt.figure(1)
    corr_sorted.plot(kind='bar')
    plt.title("Correlation of Each Feature with the Presence of Diabetes")
    plt.ylabel("Correlation Value")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()