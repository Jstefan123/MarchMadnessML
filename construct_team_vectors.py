import pandas as pd
import json

year = '17-18'

team_df = pd.read_csv('filtered_data/' + year + '/team_season.csv')
opp_df = pd.read_csv('filtered_data/' + year + '/opp_season.csv')
ap_poll_df = pd.read_csv('filtered_data/' + year + '/ap_history.csv')
misc_df = pd.read_csv('filtered_data/' + year + '/team_misc.csv')

team_ids = {}
with open('filtered_data/team_ids.json') as infile:
    team_ids = json.load(infile)


team_vectors = {}

for index, row in team_df.iterrows():

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]] = data

for index, row in opp_df.iterrows():

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]] += data


for index, row in misc_df.iterrows():

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]] += data


ap_data = {}
for index, row in ap_poll_df.iterrows():

    data = list(row[1:-1])

    # handle data inconsistencies
    if 'Saint Mary\'s' in row['Team']:
        team_vectors[team_ids['Saint Mary\'s (CA)']].append(data)

    elif 'TCU' in row['Team']:
        team_vectors[team_ids['Texas Christian']].append(data)

    elif 'UNC' in row['Team']:
        team_vectors[team_ids['North Carolina']].append(data)

    elif 'USC' in row['Team']:
        team_vectors[team_ids['Southern California']].append(data)

    else:
        ap_data[team_ids[row['Team']]] = data



zero_fill = []
for i in range(ap_poll_df.shape[1] - 1):
    zero_fill.append(0)

for id in team_vectors:

    if id in ap_data:
        team_vectors[id] += ap_data[id]
    else:
        team_vectors[id] += zero_fill



with open('filtered_data/17-18/team_vectors.json', 'w') as outfile:
    json.dump(team_vectors, outfile)
