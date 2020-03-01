import pandas as pd
import json
import os

tourney_Seeds = pd.read_csv('raw_data/tourney_seeds.csv')

curr_year = 1984
PATH_ROOT = 'filtered_data/'
new_file = ''

seeds = {"Midwest" : {}, "East": {}, "South": {}, "West": {}}

for index, row in tourney_Seeds.iterrows():

    if row['Season'] != curr_year:

        # dont write anything for the first year
        if curr_year != 1984:

            print(curr_year + 1)

            with open(new_file, 'w') as outfile:
                json.dump(seeds, outfile)

            seeds = {"Midwest" : {}, "East": {}, "South": {}, "West": {}}

        curr_year += 1
        path = PATH_ROOT + str(curr_year)[2:] + '-' + str(curr_year + 1)[2:]

        if not os.path.exists(path):
            os.mkdir(path)


        new_file = path + '/tourney_seeds.json'


    region = ''

    if row['Seed'][0] == 'W':
        region = 'Midwest'
    elif row['Seed'][0] == 'X':
        region = 'East'
    elif row['Seed'][0] == 'Y':
        region = 'South'
    else:
        region = 'West'

    try:
        seeds[region][int(row['Seed'][1:])] = row['TeamID']
    # dont need to cast to int because play in game, seed is '12a/b'
    except ValueError:
        seeds[region][row['Seed'][1:]] = row['TeamID']


# write the last years results
print(curr_year + 1)
with open(new_file, 'w') as outfile:
    json.dump(seeds, outfile)
