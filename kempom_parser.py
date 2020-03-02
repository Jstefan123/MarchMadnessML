import pandas as pd
import re

RAW_DATA_PATH_ROOT = 'data/raw_data/'
FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'

year = '18-19'
raw_file = RAW_DATA_PATH_ROOT + year + '/kempom.txt'
filtered_csv = FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.csv'

# eliminate the ncaa seed ranking in each row
with open(raw_file, 'r') as kempom_raw:

    lines = kempom_raw.readlines()

    # the only columns we are going to keep are rank and team name
    with open (filtered_csv, 'w+') as outfile:

        # for the first line, just put commas in between heading
        header = lines[0].split('\t')[:2]
        outfile.write(','.join(header) + '\n')

        # now for each line place commas in between but need to elimiate number after team
        # name which represent tourney seed for schools in tournament
        for i in range(1, len(lines)):

            # need to skip every 42nd and 43rd lines (2 mid page headers)
            if i != 1 and (i % 42 == 0 or (i + 1) % 42 == 0):
                continue

            curr_line = lines[i].split('\t')[:2]
            # this number is appended in the team name col (2nd entry)
            match_obj = re.search('[0-9]', curr_line[1])

            if match_obj is not None:
                # this is the first index of a number (remove up to the spot before
                # this because we want to remove space in front of this num)
                index = match_obj.start()
                curr_line[1] = curr_line[1][:index - 1]

            # unsplit object and write to file
            outfile.write(','.join(curr_line) + '\n')



    #for i in range(1, len(lines))