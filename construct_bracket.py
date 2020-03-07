import pandas as pd
import numpy as np
import json

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif, mutual_info_classif
from collections import Counter

FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'


# ========= lr: 0.1 max_d: 4 max_f: 0.5 num_trees: 250 =========
# mean accuracy: 0.7376831907929469

def filter_features(X_train, Y_train, ptile, team_vectors):

    # Keep ptile% best performing features (filter approach feature selection)
    selector = SelectPercentile(f_classif, percentile=ptile).fit(X_train, Y_train)
    X_train = selector.transform(X_train)

    # remove features from every feature vector in the test space
    for id in team_vectors:
        team_vectors[id] = selector.transform(np.array(team_vectors[id]).reshape(1, -1))[0]

    return X_train, team_vectors

# this can only be used for brackets 2011-present based on the first four format
def load_model(X_train, Y_train):

    return GradientBoostingClassifier(max_features=.80,
                                     max_depth=1,
                                     n_estimators=1000,
                                     learning_rate=0.075
                                     ).fit(X_train, Y_train)

def load_year_data(year):

    team_names = {}
    team_vectors = {}
    tourney_seeds = {}

    with open(FILTERED_DATA_PATH_ROOT + year + '/team_vectors.json') as infile:
        team_vectors = json.load(infile)

    with open(FILTERED_DATA_PATH_ROOT + year + '/tourney_seeds.json') as infile:
        tourney_seeds = json.load(infile)

    with open(FILTERED_DATA_PATH_ROOT + 'team_names.json') as infile:
        team_names = json.load(infile)

    return team_names, team_vectors, tourney_seeds


def create_matchup_vector(team_vectors, team1_id, team2_id):
    return np.subtract(team_vectors[str(team1_id)], team_vectors[str(team2_id)])



def print_matchup(team1, team2, winner, team_names):

    print('(' + team1[0] + ') ' + team_names[team1[1]])
    print('\t(' + winner[0] + ') ' + team_names[winner[1]])
    print('(' + team2[0] + ') ' + team_names[team2[1]])

    print('\n', end='')


# cant use normal format for matchups becuase this is a bit obscure, results just
# replace the spot in tourney seeds that the first four team won
def first_four(clf, team_names, team_vectors, tourney_seeds):

    num_features = len(list(team_vectors.values())[0])

    matchups = {}

    for region in tourney_seeds.keys():

        # if there is a letter in one of the seeds its a play in game
        # only need to check higher seeds
        for s in range(11, 17):

            try:
                tourney_seeds[region][str(s)]
            except KeyError:

                if region not in matchups:
                    matchups[region] = {}

                matchups[region][s] = [tourney_seeds[region][str(s) + 'a'], tourney_seeds[region][str(s) + 'b']]

    # construct a X_test and Y_test for these matchups

    X_test = np.zeros((4, num_features))
    matchup_num = 0

    for region in matchups:
        for s in matchups[region]:
            team1_id = matchups[region][s][0]
            team2_id = matchups[region][s][1]
            X_test[matchup_num, :] = create_matchup_vector(team_vectors, team1_id, team2_id)
            matchup_num += 1

    y_pred = clf.predict(X_test)

    # print the results in a readable format
    print("====================" + "==========" + "====================")
    print("====================" + "First Four" + "====================")

    matchup_num = 0
    for region in matchups:
        for s in matchups[region]:


            print('(' + str(s) + ')' + team_names[str(matchups[region][s][0])])

            winner = ''
            # if team1 won
            if y_pred[matchup_num] == 1:
                winner = str(matchups[region][s][0])
            # else team2 won
            else:
                winner = str(matchups[region][s][1])
            print('\t\t(' + str(s) + ') ' + team_names[winner])


            print('(' + str(s) + ')' + team_names[str(matchups[region][s][1])])

            matchup_num += 1
            print('------------------------------\n', end='')

            # insert this winner into the tourney seeds
            del tourney_seeds[region][str(s) + 'a']
            del tourney_seeds[region][str(s) + 'b']
            tourney_seeds[region][str(s)] = winner


    return tourney_seeds


def evaluate_matchups(X_train, Y_train, matchups, team_vectors, team_names, include_header=True):

    num_features = len(list(team_vectors.values())[0])

    # create the X_test for all matchups
    num_regions = len(list(matchups.keys()))
    matchups_per_region = len(list(matchups.values())[0])
    X_test = np.zeros((num_regions * matchups_per_region, num_features))

    num_matchups = 0
    for r in matchups:
        for m in range(matchups_per_region):

            team1_id = matchups[r][m][0][1]
            team2_id = matchups[r][m][1][1]
            X_test[num_matchups, :] = create_matchup_vector(team_vectors, team1_id, team2_id)

            num_matchups += 1

    # train 10 classifiers, each making a prediction for each matachup, take majority
    all_preds = None
    for i in range(5):

        clf = load_model(X_train, Y_train)
        curr_pred = clf.predict(X_test)

        if all_preds is None:
            all_preds = curr_pred
        else:
            all_preds = np.vstack((all_preds, curr_pred))

    # consolidate into a single dimensional array by taking the majority prediction
    y_pred = np.zeros(num_matchups)
    for i in range(num_matchups):

        col = all_preds[:, i]
        most_common = Counter(col).most_common(1)
        y_pred[i] = most_common[0][0]
        print(most_common)


    num_matchups = 0

    for r in matchups:

        if include_header:
            print("===========" + r + "===========")

        for m in range(matchups_per_region):

            if y_pred[num_matchups] == 1:
                winner = matchups[r][m][0]
            else:
                winner = matchups[r][m][1]

            print_matchup(matchups[r][m][0], matchups[r][m][1], winner, team_names)
            matchups[r][m] = winner

            num_matchups += 1

    return matchups


def evaluate_round(X_train, Y_train, team_names, team_vectors, matchups, include_header=True):

    # construct a matchup dictionary
    # matchups[region][matchup#] = [team_id1][team_id2]
    results = evaluate_matchups(X_train, Y_train, matchups, team_vectors, team_names, include_header)

    # format results for next matchup
    matchups = {}

    if 'National Championship' in results:
        return

    if 'Final Four' in results:
        matchups['National Championship'] = []
        matchups['National Championship'].append((results['Final Four'][0], results['Final Four'][1]))

    # if this was the Elite8, then we need to consolidate into single Final 4 region
    elif len(results['East']) == 1:
        matchups['Final Four'] = []
        matchups['Final Four'].append((results['East'][0], results['West'][0]))
        matchups['Final Four'].append((results['South'][0], results['Midwest'][0]))

    else:

        for region in results:

            matchups[region] = []
            for i in range(0, len(results[region]), 2):
                matchups[region].append((results[region][i], results[region][i+1]))

    return matchups

def get_tourney_team_vecs(team_vectors, tourney_seeds):

    dict = {}

    for region in tourney_seeds:
        for seed in tourney_seeds[region]:

            id = str(tourney_seeds[region][seed])

            s_int = None
            # if play in seed
            if len(seed) > 2:
                s_int = int(seed[:2])
            else:
                s_int = int(seed)

            dict[id] = team_vectors[id] + [s_int]

    return dict


def main():

    X_train = np.load('training_data/X_train.npy')
    Y_train = np.load('training_data/Y_train.npy')

    team_names, team_vectors, tourney_seeds = load_year_data('17-18')

    tourney_team_vecs = get_tourney_team_vecs(team_vectors, tourney_seeds)

    X_train, tourney_team_vecs = filter_features(X_train, Y_train, 90, tourney_team_vecs)

    # this does not really matter so im gonna keep a single prediction for these
    clf = load_model(X_train, Y_train)
    tourney_seeds = first_four(clf, team_names, tourney_team_vecs, tourney_seeds)

    matchups = {}
    for region in tourney_seeds:

        matchups[region] = []
        for s in [1, 8, 5, 4, 6, 3, 7, 2]:

            low_seed = str(tourney_seeds[region][str(s)])
            high_seed = str(tourney_seeds[region][str(17 - s)])
            matchups[region].append(((str(s), low_seed), (str(17 - s), high_seed)))

    print("===================" + "============" + "===================")
    print("===================" + "Second Round" + "===================")

    next_matchups = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, matchups)

    print("===================" + "===========" + "===================")
    print("===================" + "Third Round" + "===================")

    next_matchups = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, next_matchups)

    print("===================" + "=============" + "===================")
    print("===================" + "Sweet Sixteen" + "===================")

    next_matchups = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, next_matchups)

    print("==================" + "===========" + "==================")
    print("==================" + "Elite Eight" + "==================")

    next_matchups = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, next_matchups)

    print("==================" + "==========" + "==================")
    print("==================" + "Final Four" + "==================")

    next_matchups = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, next_matchups, False)

    print("=============" + "=====================" + "============")
    print("=============" + "National Championship" + "============")

    national_champ = evaluate_round(X_train, Y_train, team_names, tourney_team_vecs, next_matchups, False)



if __name__ == '__main__':
    main()
