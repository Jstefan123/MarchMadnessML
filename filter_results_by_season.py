import pandas as pd
import json
import os

reg_season_results = pd.read_csv('data/raw_data/regular_season_results.csv')

curr_year = 1984
PATH_ROOT = 'data/filtered_data/'
new_file = ''

results = []

for index, row in reg_season_results.iterrows():

    if row['Season'] != curr_year:

        # dont write anything for the first year
        if curr_year != 1984:
            print("reg season", curr_year)
            cols = ["WTeamID", "LTeamID", "MOV", "WLoc"]
            pd.DataFrame(results, columns=cols).to_csv(new_file, index=False)
            results = []

        curr_year += 1
        path = PATH_ROOT + str(curr_year)[2:] + '-' + str(curr_year + 1)[2:]

        if not os.path.exists(path):
            os.mkdir(path)


        new_file = path + '/reg_season_results.csv'

    data = []
    data.append(row['WTeamID'])
    data.append(row['LTeamID'])
    data.append(int(row['WScore']) - int(row['LScore']))
    data.append(row['WLoc'])
    results.append(data)


# write the last years results
print("reg season", curr_year)
cols = ["WTeamID", "LTeamID", "MOV", "WLoc"]
pd.DataFrame(results, columns=cols).to_csv(new_file, index=False)
results = []



# now do tourney results
curr_year = 1984
PATH_ROOT = 'data/filtered_data/'
new_file = ''
tourney_results = pd.read_csv('data/raw_data/ncaa_tourney_results.csv')


for index, row in tourney_results.iterrows():

    if row['Season'] != curr_year:

        # dont write anything for the first year
        if curr_year != 1984:
            print("tourney", curr_year)
            cols = ["WTeamID", "LTeamID", "MOV", "WLoc"]
            pd.DataFrame(results, columns=cols).to_csv(new_file, index=False)
            results = []

        curr_year += 1
        path = PATH_ROOT + str(curr_year)[2:] + '-' + str(curr_year + 1)[2:]

        if not os.path.exists(path):
            os.mkdir(path)


        new_file = path + '/ncaa_tourney_results.csv'

    data = []
    data.append(row['WTeamID'])
    data.append(row['LTeamID'])
    data.append(int(row['WScore']) - int(row['LScore']))
    data.append(row['WLoc'])
    results.append(data)

# write the last years results
print("tourney", curr_year)
cols = ["WTeamID", "LTeamID", "MOV", "WLoc"]
pd.DataFrame(results, columns=cols).to_csv(new_file, index=False)
