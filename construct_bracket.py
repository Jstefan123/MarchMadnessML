import pandas as pd
import numpy as np
import json

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2

# this can only be used for brackets 2011-present based on the first four format
clf = None

def load_model():

    clf = SVC(gamma="auto", kernel="linear", C=0.1, class_weight="balanced")
    X_train = np.load('training_data/X_train.npy')
    Y_train = np.load('training_data/Y_train.npy')
    clf.fit(X_train, Y_train)


def create_matchup_vector(team1_id, team2_id):
    return np.subtract(team_vectors[str(team1_id)], team_vectors[str(team2_id)])


def first_four(year):

    team_vectors = {}
    with open('filtered_data/' + year + '/team_vectors.json') as infile:
        team_vectors = json.load(infile)
    num_features = len(list(team_vectors.values())[0])

    matchups = {}
    with open('filtered_data/' + year + '/tourney_seeds.json') as infile:
        seeds = json.load(infile)

    for region in seeds.keys():

        # if there is a letter in one of the seeds its a play in game
        # only need to check higher seeds
        for s in range(11, 17):

            try:
                seeds[region][str(s)]
            except KeyError:

                if region not in matchups:
                    matchups[region] = {}

                matchups[region][s] = [seeds[region][str(s) + 'a'], seeds[region][str(s) + 'b']]


    # construct a X_test and Y_test for these matchups

    X_test = np.zeros((4, num_features))
    matchup_num = 0

    for region in matchups:
        for s in matchups[region]:
            team1_id = matchups[region][s][0]
            team2_id = matchups[region][s][1]
            X_test[matchup_num, :] = create_matchup_vector(team1_id, team2_id)
            matchup_num += 1


    y_pred = clf.predict(X_test)

    # print the results in a readable format
    print("====================" + "==========" + "====================")
    print("====================" + "First Four" + "====================")

    matchup_num = 0
    for region in matchups:
        for s in matchups[region]:

            print('(' + s + ')' + team_names[matchups[region][s][0]])

            # if team1 won
            if y_pred[matchup_num] == 1:
                print('(' + s + ')' + team_names[matchups[region][s][0]])
            # else team2 won
            else:
                print('(' + s + ')' + team_names[matchups[region][s][1]])

            print('(' + s + ')' + team_names[matchups[region][s][1]])

            matchup_num += 1

    print('\n', end='')



def second_round(year):
    # read in the seed list
    seeds = {}
    team_names = {}

    with open('filtered_data/' + year + '/tourney_seeds.json') as infile:
        seeds = json.load(infile)


    with open('filtered_data/team_names.json') as infile:
        team_names = json.load(infile)

    # print the blank bracket
    for region in seeds.keys():

        print("===========" + region + "===========")

        low_seed_order = [1, 8, 5, 4, 6, 3, 7, 2]
        for s in low_seed_order:

            low_seed = str(seeds[region][str(s)])

            play_in = False
            high_seed1 = -1
            high_seed2 = -1

            try:
                high_seed1 = str(seeds[region][str(17 - s)])
            except KeyError:
                play_in = True
                high_seed1 = str(seeds[region][str(17 - s) + 'a'])
                high_seed2 = str(seeds[region][str(17 - s) + 'b'])


            print('(' + str(s) + ')' + team_names[low_seed])
            print('(' + str(17 - s) + ')' + team_names[high_seed1], end='')

            if play_in:
                print('/' + team_names[high_seed2], end='')

            print('\n')

        print('\n', end='')


# def first_round_matchups(year):
#
#     # create X_test and X_train






def main():

    load_model()
    first_four('18-19')
















if __name__ == '__main__':
    main()
