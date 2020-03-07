import pandas as pd
import json
import os

RAW_DATA_PATH_ROOT = 'data/raw_data/'
FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'

def filter_results_by_season(df, filename, cols):

    # set curr year and make directory adn file to save
    curr_year = df['Season'][0]

    curr_dir = FILTERED_DATA_PATH_ROOT + str(curr_year - 1)[2:] + '-' + str(curr_year)[2:]
    if not os.path.exists(curr_dir):
        os.mkdir(curr_dir)
    curr_file = curr_dir + '/' + filename

    results  = []

    for index, row in df.iterrows():

        # when season changes, write curr season and move to next season
        if row['Season'] != curr_year:

            pd.DataFrame(results, columns=cols).to_csv(curr_file, index=False)
            print(curr_year)

            results = []
            curr_year = row['Season']

            curr_dir = FILTERED_DATA_PATH_ROOT + str(curr_year - 1)[2:] + '-' + str(curr_year)[2:]
            if not os.path.exists(curr_dir):
                os.mkdir(curr_dir)

            curr_file = curr_dir + '/' + filename

        data = []

        data.append(row['WTeamID'])
        data.append(row['LTeamID'])
        data.append(row['WLoc'])
        results.append(data)


    # write the last years results
    print(curr_year)
    pd.DataFrame(results, columns=cols).to_csv(curr_file, index=False)


def filter_tourney_seeds(df):

    curr_year = df['Season'][0]

    curr_dir = FILTERED_DATA_PATH_ROOT + str(curr_year - 1)[2:] + '-' + str(curr_year)[2:]
    if not os.path.exists(curr_dir):
        os.mkdir(curr_dir)

    curr_file = curr_dir + '/tourney_seeds.json'

    seeds_dict = {"East" : {}, "Midwest": {}, "West": {}, "South": {}}

    for index, row in df.iterrows():

        # update year and file when season changes
        if row['Season'] != curr_year:

            print(curr_year)

            with open(curr_file, 'w') as outfile:
                json.dump(seeds_dict, outfile)

            # reset seed dictionary
            seeds_dict = {"East" : {}, "Midwest": {}, "West": {}, "South": {}}

            curr_year += 1

            curr_dir = FILTERED_DATA_PATH_ROOT + str(curr_year - 1)[2:] + '-' + str(curr_year)[2:]
            if not os.path.exists(curr_dir):
                os.mkdir(curr_dir)

            curr_file = curr_dir + '/tourney_seeds.json'


        region = ''

        if row['Seed'][0] == 'W':
            region = 'East'
        elif row['Seed'][0] == 'X':
            region = 'West'
        elif row['Seed'][0] == 'Y':
            region = 'Midwest'
        else:
            region = 'South'

        try:
            seeds_dict[region][int(row['Seed'][1:])] = row['TeamID']
        # dont need to cast to int because play in game, seed is '12a/b'
        except ValueError:
            seeds_dict[region][row['Seed'][1:]] = row['TeamID']

    # write last years results
    print(curr_year)
    with open(curr_file, 'w') as outfile:
        json.dump(seeds_dict, outfile)


def main():

    reg_season_results = pd.read_csv(RAW_DATA_PATH_ROOT + 'regular_season_results.csv')
    tourney_results = pd.read_csv(RAW_DATA_PATH_ROOT + 'ncaa_tourney_results.csv')
    tourney_seeds = pd.read_csv(RAW_DATA_PATH_ROOT + 'tourney_seeds.csv')


    cols = ["WTeamID", "LTeamID", "WLoc"]

    print('=========Regular Season Results=========')
    filter_results_by_season(reg_season_results, 'reg_season_results.csv', cols)
    print('=========Tourney Season Results=========')
    filter_results_by_season(tourney_results, 'ncaa_tourney_results.csv', cols)
    print('=========Tourney Seeds=========')
    filter_tourney_seeds(tourney_seeds)






if __name__ == '__main__':
    main()
