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

    # if the team has been seen in data before this grab the id, else (just
    # got promoted to D1) then give them the next id #
    if row['Team'] not in team_ids:
        team_ids[row['Team']] = len(team_ids)

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]] = data

for index, row in opp_df.iterrows():

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]].append(data)


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
        team_vectors[team_ids[row['Team']]].append(data)


for index, row in misc_df.iterrows():

    data = list(row[1:-1])
    team_vectors[team_ids[row['Team']]].append(data)


# write new update team ids
with open('filtered_data/team_ids.json', 'w') as outfile:
    json.dump(team_ids, outfile)


with open('filtered_data/17-18/team_vectors.json', 'w') as outfile:
    json.dump(team_vectors, outfile)
