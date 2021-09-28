import sys
import argparse

import json
parser = argparse.ArgumentParser()
#import normal_methods,basic_models
from normal_methods import Normalizator
from basic_models import run_basic_models

parser = argparse.ArgumentParser()
#parser.add_argument("-a", "--all", default="all_data", help="Run with all datasets")
parser.add_argument('-d', '--dataset', default='adult', help='Dataset to use')
parser.add_argument('-m', '--method', default='zscore', help='Normalization method to use')
args = parser.parse_args()


def get_datasets():
    config = json.load(open("dataset_config.json"))["datasets"]
    return (config.keys())
    
def normalize(dataset,verbose=False):
    norm = Normalizator(dataset=dataset)
    if args.method == "all":
        method = ["zscore","minmax","tanh","variablescaling"]
        for m in method:
            print("Running normalization method: {}".format(m))
            norm.normalize(method=m,save=True)
        
    else:
        norm.normalize(args.method, save=True)
    if verbose:
        print(norm.df.describe())
        print(norm.df_norm.describe())


def main():
    """
    Here we run the pipeline
        we can run with all datasets or with a specific one
    """
    datasets = get_datasets()
    if args.dataset == "all":
        for dataset in datasets:
            normalize(dataset,verbose=True)
            output = run_basic_models(dataset)
            print(output)
    else: #run for a single one
        if not args.dataset in datasets:
            print("Dataset not found")
            sys.exit(1)
        #! NORMALIZATION - MAT
        #* Write the input and output of your normalization method here
        #* Output results in output/post_norma_data with proper names (e.g wine_zscore.csv)
        normalize(args.dataset)

        #! BASIC MODELS - JOHAN
        #* Reads normalized data from output/post_norma_data in the specified format and runs classifiers
        #* Reads the unnormalized data from datasets
    
        output = run_basic_models(args.dataset) #TODO ISAK CHANGE THIS TO THE OUTPUT YOU WANT TO DISPLAY
        print(output)
    
    #! Adv MODELS - JOAO

    #! EVALUATION - ISAK
    # Write the input you want to evaluate here and the output you will produce



if __name__ == "__main__":
    main()