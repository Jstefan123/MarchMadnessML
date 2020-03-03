import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif

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


def filter_features(X_train, Y_train, X_test, ptile):

    # Keep ptile% best performing features (filter approach feature selection)
    selector = SelectPercentile(f_classif, percentile=ptile).fit(X_train, Y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    return X_train, X_test

def cv_performance_feature_trimming(clf, X, y, k=5, ptile=75, metric="accuracy"):

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

        X_train, X_test = filter_features(X_train, Y_train, X_test, ptile)
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)

        scores.append(np.float64(performance(Y_test, y_pred, metric)))

    # And return the average performance across all fold splits.
    return np.array(scores).mean()

def tune_classifier(X_train, Y_train):

    # find a linear param C
    max = -1
    max_params = []


    max_features = []
    for i in range(10, 101, 5):
        max_features.append(float(i / 100))
    max_depth = []
    for i in range(1, 7):
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


def tune_feature_space(X_train, Y_train):

    max = -1
    max_ptile = -1

    ptile_range = []

    for i in range(25, 100, 5):
        ptile_range.append(i)

    for ptile in ptile_range:

        print("===============", str(ptile) + "% features ===============")
        clf = GradientBoostingClassifier(max_features=0.25, max_depth=3)
        cv_perf = cv_performance_feature_trimming(clf, X_train, Y_train, 20, ptile)

        print("mean accuracy:", cv_perf)

        if cv_perf > max:
            max = cv_perf
            max_ptile = ptile


    print("ENDING MAX", max, max_ptile)


def tune_params(X_train, Y_train, learn_rate, max_depth, max_features, num_estimators):

    max = -1
    max_learn_rate_found = -1
    max_depth_found = -1
    max_features_found = -1
    max_trees_found = -1

    for rate in learn_rate:
        for max_d in max_depth:
            for max_f in max_features:
                for num_trees in num_estimators:

                    print("========= lr:", rate, 'max_d:', max_d, end=' ')
                    print("max_f:", max_f, "num_trees:", num_trees, "=========")

                    clf = GradientBoostingClassifier(max_features=max_f,
                                                     max_depth=max_d,
                                                     n_estimators=num_trees,
                                                     learning_rate=rate
                                                     )

                    cv_perf = cv_performance_feature_trimming(clf, X_train, Y_train, 20, 0.75)

                    print("mean accuracy:", cv_perf)

                    if cv_perf > max:
                        print("NEW MAXXXXX================================")
                        max = cv_perf
                        max_learn_rate_found = rate
                        max_depth_found = max_d
                        max_features_found = max_f
                        max_trees_found = num_trees

    print('====================FINISHED========================')
    print('max_acc =', max)
    print('learning_rate=', max_learn_rate_found)
    print('max_depth=', max_depth_found)
    print('max_features=', max_features_found)
    print('num_estimators=', max_trees_found)

def main():

    # load training data
    X_train = np.load('training_data/X_train.npy')
    Y_train = np.load('training_data/Y_train.npy')

    max_feature_range = []
    max_depth_range = []
    learning_rate_range = []
    num_estimators = []

    max_feature_range = [.50, .60, .70, .80, .90]
    max_depth_range = [2, 3, 4]
    learning_rate_range = [.05, .075, .1]
    num_estimators = [100, 250, 500]


    tune_params(X_train, Y_train, learning_rate_range, max_depth_range, max_feature_range, num_estimators)














if __name__ == '__main__':
    main()
