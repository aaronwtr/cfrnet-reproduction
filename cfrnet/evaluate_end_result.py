import argparse
import scipy.stats
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")

    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    for k in df.columns:
        data = df[k]
        interval = scipy.stats.t.interval(alpha=0.95, df=len(data) - 1,
                               loc=np.mean(data), scale=scipy.stats.sem(data))
        print(f"{k}: {interval}")

if __name__ == "__main__":
    main()