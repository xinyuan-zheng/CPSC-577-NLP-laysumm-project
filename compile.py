import pandas as pd
import glob
import json


def collect_test():
    out_data = []
    for filename in glob.glob("../test/elife/top_50-*.json"):
        with open(filename) as json_file:
            tmp = json.load(json_file)
            out_data.append(pd.DataFrame(tmp))

    if out_data:
        out_df = pd.concat(out_data, ignore_index=True)
        out_df.to_csv('../test/elifeTop50LLM.csv', index=False)
    else:
        print("No JSON files found or data is empty.")


if __name__ == '__main__':
    collect_test()