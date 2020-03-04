import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
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


def filter_features(X_train, Y_train, X_test, ptile):

    # Keep ptile% best performing features (filter approach feature selection)
    selector = SelectPercentile(f_classif, percentile=ptile).fit(X_train, Y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    return X_train, X_test

def cv_performance(clf, X, y, k=5, ptile=75, metric="accuracy"):

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


def tune_params_GradientBoost(X_train, Y_train, learn_rate, max_depth, max_features, num_estimators):

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

                    cv_perf = cv_performance(clf, X_train, Y_train, 20, 0.75)

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


def tune_params_AdaBoost(X_train, Y_train, learn_rate, num_estimators):

    max = -1
    max_learn_rate_found = -1
    max_trees_found = -1

    for rate in learn_rate:
        for num_trees in num_estimators:

            print("================= lr:", rate, 'num_trees:', num_trees, "=================")

            clf = AdaBoostClassifier(n_estimators=num_trees, learning_rate=rate)
            cv_perf = cv_performance(clf, X_train, Y_train, 10, 0.75)

            print("mean accuracy:", cv_perf)

            if cv_perf > max:
                print("NEW MAXXXXX================================")
                max = cv_perf
                max_learn_rate_found = rate
                max_trees_found = num_trees

    print('====================FINISHED========================')
    print('max_acc =', max)
    print('learning_rate=', max_learn_rate_found)
    print('num_estimators=', max_trees_found)


def main():

    # load training data
    X_train = np.load('training_data/X_train.npy')
    Y_train = np.load('training_data/Y_train.npy')

    # max_feature_range = []
    # max_depth_range = []
    # learning_rate_range = []
    # num_estimators = []
    #
    # max_feature_range = [.50, .60, .70, .80, .90]
    # max_depth_range = [2, 3, 4]
    # learning_rate_range = [.05, .075, .1]
    # num_estimators = [100, 250, 500]


    # tune_params(X_train, Y_train, learning_rate_range, max_depth_range, max_feature_range, num_estimators)
    num_trees = [300, 500, 700, 900]
    learning_rate = [0.005, 0.00625, 0.0075, 0.00875, 0.01]
    tune_params_AdaBoost(X_train, Y_train, learning_rate, num_trees)

# ================= lr: 0.005 num_trees: 300 =================
# mean accuracy: 0.7512006104328524

# ================= lr: 0.0075 num_trees: 900 =================
#  accuracy: 0.7521448390677025











if __name__ == '__main__':
    main()
