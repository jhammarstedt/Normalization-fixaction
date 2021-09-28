# Python script to run the pipeline
import sys
import argparse
import json
parser = argparse.ArgumentParser()
#import normal_methods,basic_models
from normal_methods import Normalizator
from basic_models import run_basic_models


#parser.add_argument("-a","--all",default = "all_data",help="Run with all datasets")
#args = parser.parse_args()

def get_datasets():
    config = json.load(open("dataset_config.json"))["datasets"]
    return (config.keys())


if __name__ == "__main__":
    """Here we run the pipeline
    we can run with all datasets or with a specific one
    """
    #if args.all:
    datasets = get_datasets()

    #NORMALIZATION - MAT
    #Write the input and output of your normalization method here
    # Normalize all data with the different methods
    #   or just one dataset args (to be added)
    #! Output results in output/post_norma_data with proper names (e.g wine_zscore.csv)
    


    #BASIC MODELS - JOHAN
    #* Reads normalized data from output/post_norma_data in the specified format and runs classifiers
    #* Reads the unnormalized data from datasets
    for dataset in datasets:
        output = run_basic_models(dataset)
        print(output)
    #outputs a string for now, ISAK specifies what output he wants
    


    #Adv MODELS - JOAO

    # EVALUATION - ISAK
    # Write the input you want to evaluate here and the output you will produce
    
    #data_file = sys.argv[1]
    #print(data_file)
    # print(args.all)
    #normal_methods.run_normal_methods(data_file)
    #basic_models.run_basic_models(data_file)
