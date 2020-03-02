import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2

from matplotlib import pyplot as plt
import scikitplot as skplt


def performance(y_true, y_pred, metric="accuracy"):

    # do not need confusion matrix
    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)


    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()

    if metric == "accuracy":
        # (TP + TN) / all
        acc = (TP + TN) / (TN + FP + FN + TP)
        return (TP + TN) / (TN + FP + FN + TP)

    elif metric == "f1-score":
        # (2TP) / (2TP + FP + FN)
        return (2 * TP) / ((2 * TP) + FP + FN)

    elif metric == "precision":
        # TP / (TP + FP)
        return TP / (TP + FP)

    elif metric == "sensitivity":
        # TP / (TP + FN)
        return TP / (TP + FN)

    else:
        # TN / (TN + FP)
        return TN / (TN + FP)

def cv_performance(clf, X, y, k=5, metric="accuracy"):

    # create k-folds
    skf = StratifiedKFold(n_splits=k)

    scores = []

    # loop through each k-fold
    for train_indices, test_indices in skf.split(X, y):

        # create training and testing matrices from index lists
        X_train = X[train_indices]
        Y_train = y[train_indices]
        X_test = X[test_indices]
        Y_test = y[test_indices]

        clf.fit(X_train, Y_train)

        if metric == "auroc":
            y_pred = clf.decision_function(X_test)
            scores.append(np.float64(performance(Y_test, y_pred, metric)))

        else:
            y_pred = clf.predict(X_test)
            scores.append(np.float64(performance(Y_test, y_pred, metric)))

    # And return the average performance across all fold splits.
    return np.array(scores).mean()

def main():

    # load training data
    X_train = np.load('training_data/X_train.npy')
    Y_train = np.load('training_data/Y_train.npy')

    # find a linear param C
    max = -1
    max_params = []


    max_features = []
    for i in range(10, 101, 5):
        max_features.append(float(i / 100))
    max_depth = []
    for i in range(1, 11):
        max_depth.append(i)

    for max_f in max_features:
        for max_d in max_depth:

            print("========= max_f:", max_f, "max_d:", max_d, "=========")
            clf = GradientBoostingClassifier(max_features=max_f, max_depth=max_d)
            cv_perf = cv_performance(clf, X_train, Y_train, 20, "accuracy")

            print("mean accuracy:", cv_perf)

            if cv_perf > max:
                max = cv_perf
                max_params = [max_f, max_d]

    print("ENDING MAX", max, max_params)













if __name__ == '__main__':
    main()