import pandas as pd

opp_basic_df = pd.read_csv('raw_data/17-18/other_basic.csv', header=1)
opp_advanced_df = pd.read_csv('raw_data/17-18/other_advanced.csv', header=1)

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
pd.DataFrame(condensed_arr, columns=cols).to_csv('filtered_data/17-18/opp_season.csv', index=False)
