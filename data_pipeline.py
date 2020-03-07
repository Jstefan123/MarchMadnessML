from construct_team_vectors import *
from random import randint
# this file houses all of the code that filters that data starting as raw data
# and outputting X_train and Y_train as well as the team_vectors to create
# matchup feature vectors for the test year

# years to consider for training
train_years = ['10-11','11-12','12-13','13-14','14-15','15-16','16-17','18-19']

# year to test on
test_year = ['17-18']

# path to the filtered_data years
FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'


def get_id_to_seed_dict(year):

    with open(FILTERED_DATA_PATH_ROOT + year + '/tourney_seeds.json') as infile:
        tourney_seeds = json.load(infile)

    dict = {}

    for region in tourney_seeds:
        for seed in tourney_seeds[region]:

            id = tourney_seeds[region][seed]

            s_int = None
            # if play in seed
            if len(seed) > 2:
                s_int = int(seed[:2])
            else:
                s_int = int(seed)

            dict[str(id)] = s_int

    return dict

# given winning/losing id and winning location, return the feature vector
#and true label for the matchup
def get_matchup_vector_and_label(W_vec, L_vec, W_Loc):

    home_team = None
    away_team = None
    true_y = None

    # do not create any bias from neutral site games by randomizing
    # who is home and away
    if W_Loc == 'N':

        if randint(0, 1) == 1:
            home_team = np.array(L_vec)
            away_team = np.array(W_vec)
            true_y = -1
        else:
            home_team = np.array(W_vec)
            away_team = np.array(L_vec)
            true_y = 1

    elif W_Loc == 'H':
        home_team = np.array(W_vec)
        away_team = np.array(L_vec)
        true_y = 1

    else:
        home_team = np.array(L_vec)
        away_team = np.array(W_vec)
        true_y = -1


    return np.subtract(home_team, away_team), true_y

# if only_tourny_data is True only include tournament game results
def construct_training_data(training_years, kempom_ceiling, only_tourny_data=False):

    TOTAL_X_train = None
    TOTAL_Y_train = None

    for year in training_years:

        # load in kempom rankings, results, and team vectors from this year -----------------
        # this is the last time we will need this for this year so we can delete it after
        kempom_rankings = {}
        with open(FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.json') as infile:
            kempom_rankings = json.load(infile)

        os.remove(FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.json')

        team_vectors = {}
        with open(FILTERED_DATA_PATH_ROOT + year + '/team_vectors.json') as infile:
            team_vectors = json.load(infile)

        # ------------------------------------------------------------------------------------

        X_train = None
        Y_train = None

        if not only_tourny_data:

            reg_season_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/reg_season_results.csv')
            for index, row in reg_season_df.iterrows():

                W_id = str(row['WTeamID'])
                L_id = str(row['LTeamID'])

                # if not kempom viable by our given threshold then do not consider in
                # our training data -------------------------------------------------
                kempom1 = int(kempom_rankings[W_id])
                kempom2 = int(kempom_rankings[L_id])

                if kempom1 > kempom_ceiling and kempom2 > kempom_ceiling:
                    continue

                W_vec = team_vectors[W_id]
                L_vec = team_vectors[L_id]
                matchup_vec, true_y = get_matchup_vector_and_label(W_vec, L_vec, row['WLoc'])

                # if first matchup, avoid None error
                if X_train is None:
                    X_train = matchup_vec
                    Y_train = np.array([true_y])
                else:
                    X_train = np.vstack((X_train, matchup_vec))
                    Y_train = np.concatenate([Y_train, [true_y]])

        tourney_df = pd.read_csv(FILTERED_DATA_PATH_ROOT + year + '/ncaa_tourney_results.csv')

        # if only considering tourney data we need to load in the tourney seeds but then
        # flip it so it maps team_id to regional seed
        tourney_seeds = {}
        if only_tourny_data:
            tourney_seeds = get_id_to_seed_dict(year)

        for index, row in tourney_df.iterrows():

            W_vec = team_vectors[str(row['WTeamID'])]
            L_vec = team_vectors[str(row['LTeamID'])]

            # if we are only considering tourney data we need to take on the seeds
            if only_tourny_data:
                W_vec = W_vec + [tourney_seeds[str(row['WTeamID'])]]
                L_vec = L_vec + [tourney_seeds[str(row['LTeamID'])]]


            matchup_vec, true_y = get_matchup_vector_and_label(W_vec, L_vec, row['WLoc'])

            # if first matchup, avoid None error
            if X_train is None:
                X_train = matchup_vec
                Y_train = np.array([true_y])
            else:
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


def main():

    # constructing team vectors for training and testing years
    for year in (train_years + test_year):
        filter_ap_data(year)
        filter_team_season_data(year)
        filter_opp_season_data(year)
        filter_misc_team_data(year)
        create_kempom_dict(year)
        construct_team_vectors(year)

    # consalidate training data into single matrices X_train and Y_train
    # located in training_data/
    all_years = train_years + test_year
    construct_training_data(train_years, 75, True)






if __name__ == '__main__':
    main()
