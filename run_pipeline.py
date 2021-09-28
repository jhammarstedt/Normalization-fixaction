import sys
import argparse
from normal_methods import Normalizator

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", default="all_data", help="Run with all datasets")
parser.add_argument('-d', '--dataset', default='adult', help='Dataset to use')
parser.add_argument('-m', '--method', default='zscore', help='Normalization method to use')
args = parser.parse_args()


def main():
    norm = Normalizator(dataset=args.dataset)
    norm.normalize(args.method)
    print(norm.df.describe())
    print(norm.df_norm.describe())


if __name__ == "__main__":
    main()
    #normal_methods.run_normal_methods(data_file)
    #basic_models.run_basic_models(data_file)
