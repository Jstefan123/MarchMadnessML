import pandas as pd
import numpy as np
import json
import random

# the format of our data will be the home vector - away vector
# true label will be 1 if home team wins, -1 if away team wins
# if the data shows a neutral site game, home team will be randomized

TOTAL_X_train = None
TOTAL_Y_train = None

FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'
year_range = ['15-16','16-17','17-18']

# at least one team in the matchup must have a kempom rating below this threshold
# to be considered in the training data (for an ncaa tournament we wouldnt care about
# garbage games between kempom 200-300 teams)
kempom_ceiling = 125

for year in year_range:

    kempom_rankings = {}
    with open(FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.json') as infile:
        kempom_rankings = json.load(infile)

    reg_season_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/reg_season_results.csv')
    tourney_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/ncaa_tourney_results.csv')


    team_vectors = {}
    with open(FILTERED_DATA_PATH_ROOT + year + '/team_vectors.json') as infile:
        team_vectors = json.load(infile)

    X_train = None
    Y_train = None

    for index, row in reg_season_df.iterrows():

        id1 = str(row['WTeamID'])
        id2 = str(row['LTeamID'])
        if int(kempom_rankings[id1]) > kempom_ceiling and int(kempom_rankings[id2]) > kempom_ceiling:
            continue

        home_team = []
        away_team = []
        true_y = 0

        # neutral site game, generate a random number where 0 = winning team is home
        # and 0 = away team was home
        if row['WLoc'] == 'N':

            if random.randint(0, 1) == 1:
                home_team = np.array(team_vectors[str(row['LTeamID'])])
                away_team = np.array(team_vectors[str(row['WTeamID'])])
                true_y = -1
            else:
                home_team = np.array(team_vectors[str(row['WTeamID'])])
                away_team = np.array(team_vectors[str(row['LTeamID'])])
                true_y = 1

        elif row['WLoc'] == 'H':
            home_team = np.array(team_vectors[str(row['WTeamID'])])
            away_team = np.array(team_vectors[str(row['LTeamID'])])
            true_y = 1

        else:
            home_team = np.array(team_vectors[str(row['LTeamID'])])
            away_team = np.array(team_vectors[str(row['WTeamID'])])
            true_y = -1

        matchup_vec = np.subtract(home_team, away_team)

        if X_train is None:
            X_train = matchup_vec
            Y_train = np.array([true_y])
        else:
            X_train = np.vstack((X_train, matchup_vec))
            Y_train = np.concatenate([Y_train, [true_y]])

    for index, row in tourney_df.iterrows():

        home_team = []
        away_team = []
        true_y = 0

        # neutral site game, generate a random number where 0 = winning team is home
        # and 0 = away team was home
        if row['WLoc'] == 'N':

            if random.randint(0, 1) == 1:
                home_team = np.array(team_vectors[str(row['LTeamID'])])
                away_team = np.array(team_vectors[str(row['WTeamID'])])
                true_y = -1
            else:
                home_team = np.array(team_vectors[str(row['WTeamID'])])
                away_team = np.array(team_vectors[str(row['LTeamID'])])
                true_y = 1

        elif row['WLoc'] == 'H':
            home_team = np.array(team_vectors[str(row['WTeamID'])])
            away_team = np.array(team_vectors[str(row['LTeamID'])])
            true_y = 1

        else:
            home_team = np.array(team_vectors[str(row['LTeamID'])])
            away_team = np.array(team_vectors[str(row['WTeamID'])])
            true_y = -1

        # X_train wont be None here so dont need to check
        matchup_vec = np.subtract(home_team, away_team)
        X_train = np.vstack((X_train, matchup_vec))
        Y_train = np.concatenate([Y_train, [true_y]])


    # add to the total training matrices
    if TOTAL_X_train is None:
        TOTAL_X_train = X_train
        TOTAL_Y_train = Y_train
    else:
        TOTAL_X_train = np.vstack((TOTAL_X_train, X_train))
        TOTAL_Y_train = np.concatenate([TOTAL_Y_train, Y_train])


np.save('training_data/X_train', TOTAL_X_train)
np.save('training_data/Y_train', TOTAL_Y_train)
