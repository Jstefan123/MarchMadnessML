import pandas as pd

orig_df = pd.read_csv('raw_data/17-18/ap_poll_history.csv')

condensed_arr = []

# record the following data:
#   preseason position
#   last week position
#   peak position
#   peak position week
#   weeks in top 25
#   weeks in top 10

for index, row in orig_df.iterrows():

    # skip the first 2 header rows
    if index == 0 or index == 1:
        continue

    data = []

    # team name
    data.append(row[0])
    # first week
    data.append(row[2])
    #last week
    data.append(row[len(row) - 1])

    # find peak, peak position, weeks in top 10, weeks in top 25

    weeks_in_top_25 = 0
    weeks_in_top_10 = 0
    peak_position = 26
    peak_position_week = 0

    for i in range(2, len(row)):

        if not pd.isnull(row[i]):
            weeks_in_top_25 += 1

            if int(row[i]) <= peak_position:
                peak_position = int(row[i])
                peak_position_week = i

            if int(row[i]) < 11:
                weeks_in_top_10 += 1


    data.append(peak_position)
    data.append(peak_position_week)
    data.append(weeks_in_top_25)
    data.append(weeks_in_top_10)
    condensed_arr.append(data)


cols = ['Team', "Preseason_Rank", "Last_Week_Rank", "Peak_Position", "Peak_Position_Week", "Weeks_in_Top25", "Weeks_in_Top10"]
pd.DataFrame(condensed_arr, columns=cols).to_csv('filtered_data/17-18/condensed_ap_history.csv', index=False)
