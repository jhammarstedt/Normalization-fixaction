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
    """Here we run the pipeline
    we can run with all datasets or with a specific one
    """

    #NORMALIZATION - MAT
    #Write the input and output of your normalization method here
    #! Output results in output/post_norma_data with proper names (e.g wine_zscore.csv)





    #BASIC MODELS - JOHAN
    #* Reads normalized data from output/post_norma_data in the specified format and runs classifiers
    #* Reads the unnormalized data from datasets



    #Adv MODELS - JOAO

    # EVALUATION - ISAK
    # Write the input you want to evaluate here and the output you will produce

    #data_file = sys.argv[1]
    #print(data_file)
    # print(args.all)
    #normal_methods.run_normal_methods(data_file)
    #basic_models.run_basic_models(data_file)
