import pandas as pd

conf_summary_df = pd.read_csv('raw_data/17-18/conf_summary.csv')
school_ratings_df = pd.read_csv('raw_data/17-18/school_conf_and_ratings.csv', header=1)

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

    # if MAC, dont include east, west tags
    conf = ''
    if 'MAC' in row['Conf']:
        conf = 'MAC'
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
pd.DataFrame(condensed_arr, columns=cols).to_csv('filtered_data/17-18/team_misc.csv', index=False)
