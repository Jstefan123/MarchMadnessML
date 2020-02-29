import pandas as pd

team_basic_df = pd.read_csv('raw_data/17-18/school_basic.csv', header=1)
team_advanced_df = pd.read_csv('raw_data/17-18/school_advanced.csv', header=1)

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

    data.append(row['W'])
    data.append(row['L'])

    # home wins
    data.append(row['W.2'])
    data.append(row['L.2'])

    # away wins
    data.append(row['W.3'])
    data.append(row['L.3'])

    # conference wins
    data.append(row['W.1'])
    data.append(row['L.1'])

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


cols = ["Team", "W", "L", "Home_W", "Home_L", "Away_W", "Away_L", "Conf_W", "Conf_L", "SOS", "FG", "FG%",
        "3P", "3P%", "FT", "FT%", "ORB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "Pace", "FTAR", "3PAR", "TS%",
        "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%", "FT/FGA"]
pd.DataFrame(condensed_arr, columns=cols).to_csv('filtered_data/17-18/condensed_team_season.csv', index=False)
