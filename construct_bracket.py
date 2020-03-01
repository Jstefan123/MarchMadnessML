import pandas as pd
import numpy as np
import json


def main():

    # read in the seed list
    seeds = {}
    team_names = {}

    with open('filtered_data/17-18/tourney_seeds.json') as infile:
        seeds = json.load(infile)


    with open('filtered_data/team_names.json') as infile:
        team_names = json.load(infile)

    # print the blank bracket
    for region in seeds.keys():

        print("===========" + region + "===========")

        low_seed_order = [1, 8, 5, 4, 6, 3, 7, 2]
        for s in low_seed_order:

            low_seed = str(seeds[region][str(s)])

            play_in = False
            high_seed1 = -1
            high_seed2 = -1

            try:
                high_seed1 = str(seeds[region][str(17 - s)])
            except KeyError:
                play_in = True
                high_seed1 = str(seeds[region][str(17 - s) + 'a'])
                high_seed2 = str(seeds[region][str(17 - s) + 'b'])


            print('(' + str(s) + ')' + team_names[low_seed])
            print('(' + str(17 - s) + ')' + team_names[high_seed1], end='')

            if play_in:
                print('/' + team_names[high_seed2], end='')

            print('\n')

        print('\n', end='')
















if __name__ == '__main__':
    main()
