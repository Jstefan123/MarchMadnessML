import pandas as pd
import numpy as np
import json
import os
from kempom_parser import create_kempom_dict
from sklearn.preprocessing import normalize

RAW_DATA_PATH_ROOT = 'data/raw_data/'
FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'

def filter_ap_data(year):
    orig_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/ap_poll_history.csv')

    condensed_arr = []

    # record the following data:
    #   weeks in top 25
    #   weeks in top 10
    #   weeks in top 5
    #   weeks at 1

    for index, row in orig_df.iterrows():

        # skip the first 2 header rows
        if index == 0 or index == 1:
            continue

        data = []

        # team name
        data.append(row[0])

        # find peak, peak position, weeks in top 10, weeks in top 25

        weeks_in_top_25 = 0
        weeks_in_top_10 = 0
        weeks_in_top_5 = 0
        weeks_at_1 = 0

        for i in range(2, len(row)):

            if not pd.isnull(row[i]) and row[i] is not '-':
                weeks_in_top_25 += 1

                if int(row[i]) < 11:
                    weeks_in_top_10 += 1

                    if int(row[i]) < 6:
                        weeks_in_top_5 += 1

                        if int(row[i]) == 1:
                            weeks_at_1 += 1


        data.append(weeks_in_top_25)
        data.append(weeks_in_top_10)
        data.append(weeks_in_top_5)
        data.append(weeks_at_1)

        condensed_arr.append(data)


    cols = ['Team', "Weeks_in_Top25", "Weeks_in_Top10", "Weeks_in_Top5", "Weeks_at_1"]
    pd.DataFrame(condensed_arr, columns=cols).to_csv(FILTERED_DATA_PATH_ROOT + year + '/ap_history.csv', index=False)

def filter_team_season_data(year):
    team_basic_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/school_basic.csv', header=1)
    team_advanced_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/school_advanced.csv', header=1)

    condensed_arr = []

    # exract the following from basic data:
    # overall wins
    # overall losses
    # overall home wins
    # overall home losses
    # overall away wins
    # overall away losses
    # conference wins
    # conference losses
    # SOS (strength of schedule)
    # field goals
    # field goal percentage
    # three pointers
    # three pointer percentage
    # free throws
    # free throw percentage
    # offensive rebounds
    # total rebounds
    # total assists
    # total steals
    # total blocks
    # total turnovers
    # total personal fouls


    # extract the following from advanced data:
    # pace
    # FTAR (free throw attempt rate)
    # 3PAR (three point attempt rate)
    # TS% (true shooting percentage)
    # TRB% (total rebound percentage)
    # AST% (assist percentage)
    # STL% (steal percentage)
    # BLK% (block percentage)
    # eFG% (effective field goal percentage)
    # TOV% (turnover percentage)
    # ORB% (offensive rebound percentage)
    # FT/FGA (free throws per field goal attempt)



    for index, row in team_basic_df.iterrows():

        data = []

        if "NCAA" in row['School']:
            data.append(row['School'][:-5])
        else:
            data.append(row['School'])

        # data.append(row['W'])
        # data.append(row['L'])
        #
        # # home wins
        # data.append(row['W.2'])
        # data.append(row['L.2'])
        #
        # # away wins
        # data.append(row['W.3'])
        # data.append(row['L.3'])
        #
        # # conference wins
        # data.append(row['W.1'])
        # data.append(row['L.1'])

        data.append(row['SOS'])

        data.append(row['FG'])
        data.append(row['FG%'])
        data.append(row['3P'])
        data.append(row['3P%'])
        data.append(row['FT'])
        data.append(row['FT%'])
        data.append(row['ORB'])
        data.append(row['TRB'])
        data.append(row['AST'])
        data.append(row['STL'])
        data.append(row['BLK'])
        data.append(row['TOV'])
        data.append(row['PF'])

        condensed_arr.append(data)


    for index, row in team_advanced_df.iterrows():

        condensed_arr[index].append(row['Pace'])
        condensed_arr[index].append(row['FTr'])
        condensed_arr[index].append(row['3PAr'])
        condensed_arr[index].append(row['TS%'])
        condensed_arr[index].append(row['TRB%'])
        condensed_arr[index].append(row['AST%'])
        condensed_arr[index].append(row['STL%'])
        condensed_arr[index].append(row['BLK%'])
        condensed_arr[index].append(row['eFG%'])
        condensed_arr[index].append(row['TOV%'])
        condensed_arr[index].append(row['ORB%'])
        condensed_arr[index].append(row['FT/FGA'])

# "W", "L", "Home_W", "Home_L", "Away_W", "Away_L", "Conf_W", "Conf_L"
    cols = ["Team", "SOS", "FG", "FG%",
            "3P", "3P%", "FT", "FT%", "ORB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "Pace", "FTAR", "3PAR", "TS%",
            "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%", "FT/FGA"]
    pd.DataFrame(condensed_arr, columns=cols).to_csv(FILTERED_DATA_PATH_ROOT + year + '/team_season.csv', index=False)

def filter_opp_season_data(year):
    opp_basic_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/other_basic.csv', header=1)
    opp_advanced_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/other_advanced.csv', header=1)

    condensed_arr = []

    # exract the following from basic data:
    # field goals
    # field goal percentage
    # three pointers
    # three pointer percentage
    # free throws
    # free throw percentage
    # offensive rebounds
    # total rebounds
    # total assists
    # total steals
    # total blocks
    # total turnovers
    # total personal fouls


    # extract the following from advanced data:
    # pace
    # OR (offensive rating)
    # FTAR (free throw attempt rate)
    # 3PAR (three point attempt rate)
    # TS% (true shooting percentage)
    # TRB% (total rebound percentage)
    # AST% (assist percentage)
    # STL% (steal percentage)
    # BLK% (block percentage)
    # eFG% (effective field goal percentage)
    # TOV% (turnover percentage)
    # ORB% (offensive rebound percentage)
    # FT/FGA (free throws per field goal attempt)



    for index, row in opp_basic_df.iterrows():

        data = []

        if "NCAA" in row['School']:
            data.append(row['School'][:-5])
        else:
            data.append(row['School'])

        data.append(row['FG'])
        data.append(row['FG%'])
        data.append(row['3P'])
        data.append(row['3P%'])
        data.append(row['FT'])
        data.append(row['FT%'])
        data.append(row['ORB'])
        data.append(row['TRB'])
        data.append(row['AST'])
        data.append(row['STL'])
        data.append(row['BLK'])
        data.append(row['TOV'])
        data.append(row['PF'])

        condensed_arr.append(data)


    for index, row in opp_advanced_df.iterrows():

        condensed_arr[index].append(row['Pace'])
        condensed_arr[index].append(row['ORtg'])
        condensed_arr[index].append(row['FTr'])
        condensed_arr[index].append(row['3PAr'])
        condensed_arr[index].append(row['TS%'])
        condensed_arr[index].append(row['TRB%'])
        condensed_arr[index].append(row['AST%'])
        condensed_arr[index].append(row['STL%'])
        condensed_arr[index].append(row['BLK%'])
        condensed_arr[index].append(row['eFG%'])
        condensed_arr[index].append(row['TOV%'])
        condensed_arr[index].append(row['ORB%'])
        condensed_arr[index].append(row['FT/FGA'])


    # pace
    # OR (offensive rating)
    # FTAR (free throw attempt rate)
    # 3PAR (three point attempt rate)
    # TS% (true shooting percentage)
    # TRB% (total rebound percentage)
    # AST% (assist percentage)
    # STL% (steal percentage)
    # BLK% (block percentage)
    # eFG% (effective field goal percentage)
    # TOV% (turnover percentage)
    # ORB% (offensive rebound percentage)
    # FT/FGA (free throws per field goal attempt)
    cols = ["Team", "FG", "FG%", "3P", "3P%", "FT", "FT%", "ORB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "Pace", "OR", "FTAR", "3PAR", "TS%",
            "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%", "FT/FGA"]
    pd.DataFrame(condensed_arr, columns=cols).to_csv(FILTERED_DATA_PATH_ROOT + year + '/opp_season.csv', index=False)

def filter_misc_team_data(year):
    conf_summary_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/conf_summary.csv')
    school_ratings_df = pd.read_csv(RAW_DATA_PATH_ROOT + year + '/school_conf_and_ratings.csv', header=1)

    condensed_arr = []


    # extract following from conf_summary:
    # rank of conference (by SRS)
    # regular season champ
    # tournamnet champ


    # exract the following from school_conf_ratings:
    # Team
    # PPG
    # OPP_PPG
    # OSRS (offensive simple rating system)
    # DSRS (defensive simple rating system)
    # ORtg (offensive rating)
    # DRtg (defensive rating)

    # need to hardcode shortened names for the conferences so data matches up
    conf_names = {
    'Big 12 Conference': 'Big 12',
    'Big East Conference': 'Big East',
    'Atlantic Coast Conference': 'ACC',
    'Big Ten Conference': 'Big Ten',
    'Southeastern Conference': 'SEC',
    'Pacific-12 Conference': 'Pac-12',
    'American Athletic Conference': 'AAC',
    'Mountain West Conference': 'MWC',
    'Missouri Valley Conference': 'MVC',
    'Atlantic 10 Conference': 'A-10',
    'Conference USA': 'CUSA',
    'Mid-American Conference': 'MAC',
    'Summit League': 'Summit',
    'Sun Belt Conference': 'Sun Belt',
    'Colonial Athletic Association': 'CAA',
    'Big Sky Conference': 'Big Sky',
    'Western Athletic Conference': 'WAC',
    'Big West Conference': 'Big West',
    'Ivy League': 'Ivy',
    'Ivy Group': 'Ivy',
    'Metro Atlantic Athletic Conference': 'MAAC',
    'Southern Conference': 'Southern',
    'Ohio Valley Conference': 'OVC',
    'America East Conference': 'AEC',
    'Horizon League': 'Horizon',
    'Big South Conference': 'Big South',
    'Patriot League': 'Patriot',
    'Atlantic Sun Conference': 'A-Sun',
    'Southland Conference': 'Southland',
    'Northeast Conference': 'NEC',
    'Southwest Athletic Conference': 'SWAC',
    'West Coast Conference': 'WCC',
    'Mid-Eastern Athletic Conference': 'MEAC',
    'Independent': 'Ind',
    'Great West Conference': 'GWC'
    }


    conf_dict = {}

    for index, row in conf_summary_df.iterrows():
        conf_data = {}

        conf_data['rank'] = row['Rk']
        conf_data['season_champ'] = row['Regular Season Champ']
        conf_data['tourney_champ'] = row['Tournament Champ']
        conf_dict[conf_names[row['Conference']]] = conf_data


    for index, row in school_ratings_df.iterrows():

        data = []
        data.append(row['School'])
        data.append(row['Pts'])
        data.append(row['Opp'])
        data.append(row['OSRS'])
        data.append(row['DSRS'])
        data.append(row['ORtg'])
        data.append(row['DRtg'])

        # if MAC, OVC dont include east, west tags in (...)
        conf = ''
        if '(' in row['Conf']:
            index = row['Conf'].find('(')
            conf = row['Conf'][:index - 1]
        else:
            conf = row['Conf']

        # conference rank
        data.append(conf_dict[conf]['rank'])

        # did win regular season
        data.append(int(row['School'] == conf_dict[conf]['season_champ']))

        # did win conf tourney
        data.append(int(row['School'] == conf_dict[conf]['tourney_champ']))

        condensed_arr.append(data)

    cols = ["Team", "PPG", "OPP_PPG", "O-SRS", "D-SRS", "ORtg", "DRtg", "Conf_Rank", "Season_Conf_Champs?", "Tourney_Conf_Champs?"]
    pd.DataFrame(condensed_arr, columns=cols).to_csv(FILTERED_DATA_PATH_ROOT + year + '/team_misc.csv', index=False)


def construct_team_vectors(year):
    team_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/team_season.csv')
    opp_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/opp_season.csv')
    ap_poll_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/ap_history.csv')
    misc_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/team_misc.csv')


    # load kempom rankings
    kempom_rankings = {}
    with open(FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.json') as infile:
        kempom_rankings = json.load(infile)

    team_ids = {}
    with open(FILTERED_DATA_PATH_ROOT + 'team_ids.json') as infile:
        team_ids = json.load(infile)

    team_vectors = {}

    # first element is the kempom ranking
    for team_id in kempom_rankings:
        team_vectors[int(team_id)] = [int(kempom_rankings[team_id])]

    for index, row in team_df.iterrows():

        data = list(row[1:])
        team_vectors[team_ids[row['Team']]] += data


    for index, row in opp_df.iterrows():

        data = list(row[1:])
        team_vectors[team_ids[row['Team']]] += data


    for index, row in misc_df.iterrows():

        data = list(row[1:])
        team_vectors[team_ids[row['Team']]] += data

    ap_data = {}
    for index, row in ap_poll_df.iterrows():

        data = list(row[1:])

        # handle data inconsistencies
        if 'Saint Mary\'s' in row['Team']:
            ap_data[team_ids['Saint Mary\'s (CA)']] = data

        elif 'TCU' in row['Team']:
            ap_data[team_ids['Texas Christian']] = data

        elif 'UNC' in row['Team']:
            ap_data[team_ids['North Carolina']] = data

        elif 'USC' in row['Team']:
            ap_data[team_ids['Southern California']] = data

        else:
            ap_data[team_ids[row['Team']]] = data


    for id in team_vectors:

        if id in ap_data:
            team_vectors[id] += ap_data[id]
        else:
            team_vectors[id] += [0,0,0,0]


    # Now that we have complete feature vectors for each team, we need to construct
    # a feature matrix so we can normalize
    id_order = []
    feature_matrix = None
    for id in team_vectors:
        id_order.append(id)

        if feature_matrix is None:
            feature_matrix = np.array(team_vectors[id])
        else:
            feature_matrix = np.vstack((feature_matrix, team_vectors[id]))


    # normalize features so everything is scaled equally
    feature_matrix = normalize(feature_matrix, axis=0)

    for i in range(np.shape(feature_matrix)[0]):
        team_vectors[id_order[i]] = list(feature_matrix[i])

    with open(FILTERED_DATA_PATH_ROOT + year + '/team_vectors.json', 'w') as outfile:
        json.dump(team_vectors, outfile)

    # remove the intermediate csv files for the filtered data
    os.remove(FILTERED_DATA_PATH_ROOT + year + '/team_season.csv')
    os.remove(FILTERED_DATA_PATH_ROOT + year + '/opp_season.csv')
    os.remove(FILTERED_DATA_PATH_ROOT + year + '/ap_history.csv')
    os.remove(FILTERED_DATA_PATH_ROOT + year + '/team_misc.csv')
