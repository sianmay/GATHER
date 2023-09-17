import pandas as pd
import numpy as np
import argparse
import os

import statistics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='The directory containing the batch files.', type=str)
    parser.add_argument('-o', '--output', help='The name of the CSV file to write to.', type=str)
    args = parser.parse_args()

    collected_arr = []
    gini_arr = []

    for root, _, files in os.walk(args.input):
        for f in files:
            if f.startswith('.'):
                continue
            with open(os.path.join(root, f)) as file:
                raw_data = file.readlines()
                #print(raw_data)
                c_entry, g_entry = [], []
                for i in range(3, len(raw_data)-2, 2):
                    point = raw_data[i]
                    #print("point:",point)
                    #if point.find('Collected')!=-1:
                     #   print("point is not None")
                    c_entry.append(int(point[(point.find('Collected:') + 11): point.find('Gini') - 1]))
                    g_entry.append(float(point[(point.find('Gini:') + 6):]))

                print(len(c_entry))
                collected_arr.append(c_entry)
                gini_arr.append(g_entry)

    c_df = pd.DataFrame(np.array(collected_arr).transpose())
    g_df = pd.DataFrame(np.array(gini_arr).transpose())

    # Stats
    c_df['mean'] = c_df.mean(axis=1)
    c_df['stddev'] = c_df.std(axis=1)

    g_df['mean'] = g_df.mean(axis=1)
    g_df['stddev'] = g_df.std(axis=1)

    c_df.to_excel(args.output + '/collected.xlsx')
    g_df.to_excel(args.output + '/gini.xlsx')


if __name__ == '__main__':
    main()
