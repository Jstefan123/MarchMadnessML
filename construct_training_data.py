import pandas as pd
import numpy as np
import json
import random

# the format of our data will be the home vector - away vector
# true label will be 1 if home team wins, -1 if away team wins
# if the data shows a neutral site game, home team will be randomized

year = '18-19'

reg_season_df = pd.read_csv('filtered_data/' + year + '/reg_season_results.csv')
tourney_df = pd.read_csv('filtered_data/' + year + '/ncaa_tourney_results.csv')


team_vectors = {}
with open('filtered_data/' + year + '/team_vectors.json') as infile:
    team_vectors = json.load(infile)


num_features = len(list(team_vectors.values())[0])
num_matchups = len(reg_season_df) + len(tourney_df)

X_train = np.zeros((num_matchups, num_features))
Y_train = np.zeros(num_matchups)


matchup_num = 0
# for index, row in reg_season_df.iterrows():
#
#
#     home_team = []
#     away_team = []
#     true_y = 0
#
#     # neutral site game, generate a random number where 0 = winning team is home
#     # and 0 = away team was home
#     if row['WLoc'] == 'N':
#
#         if random.randint(0, 1) == 1:
#             home_team = np.array(team_vectors[str(row['LTeamID'])])
#             away_team = np.array(team_vectors[str(row['WTeamID'])])
#             true_y = -1
#         else:
#             home_team = np.array(team_vectors[str(row['WTeamID'])])
#             away_team = np.array(team_vectors[str(row['LTeamID'])])
#             true_y = 1
#
#     elif row['WLoc'] == 'H':
#         home_team = np.array(team_vectors[str(row['WTeamID'])])
#         away_team = np.array(team_vectors[str(row['LTeamID'])])
#         true_y = 1
#
#     else:
#         home_team = np.array(team_vectors[str(row['LTeamID'])])
#         away_team = np.array(team_vectors[str(row['WTeamID'])])
#         true_y = -1
#
#     X_train[matchup_num, :] = np.subtract(home_team, away_team)
#     Y_train[matchup_num] = true_y
#
#     matchup_num += 1

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

    X_train[matchup_num, :] = np.subtract(home_team, away_team)
    Y_train[matchup_num] = true_y

    matchup_num += 1


np.save('training_data/X_train', X_train)
np.save('training_data/Y_train', Y_train)
