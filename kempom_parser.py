import re
import json

RAW_DATA_PATH_ROOT = 'data/raw_data/'
FILTERED_DATA_PATH_ROOT = 'data/filtered_data/'

def create_kempom_dict(year):
    raw_file = RAW_DATA_PATH_ROOT + year + '/kempom.txt'
    filtered_csv = FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.csv'

    team_ids = {}
    with open(FILTERED_DATA_PATH_ROOT + 'team_ids.json') as infile:
        team_ids = json.load(infile)

    kempom_dict = {}

    # eliminate the ncaa seed ranking in each row
    with open(raw_file, 'r') as kempom_raw:

        lines = kempom_raw.readlines()

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

            # find the teamID associated with the team name and map it
            # to the kempom ranking for this year
            kempom_dict[team_ids[curr_line[1]]] = curr_line[0]


    with open(FILTERED_DATA_PATH_ROOT + year + '/kempom_rankings.json', 'w+') as outfile:
        json.dump(kempom_dict, outfile)
