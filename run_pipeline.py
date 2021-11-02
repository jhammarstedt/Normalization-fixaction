import sys
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
# import normal_methods,basic_models
from adv_models import run_advanced_models
from normal_methods import Normalizator
from basic_models import run_basic_models
from datetime import datetime
from sys import platform
import time

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", default="all_data", help="Run with all datasets")
parser.add_argument('-d', '--dataset', default='adult', help='Dataset to use')
parser.add_argument('-m', '--method', default='zscore', help='Normalization method to use')
parser.add_argument('-s', '--seed', default=1, type=int, help='Seed')
parser.add_argument('-mm', '--model', default="all", help="Select all, basic or adv")
parser.add_argument('-bn', '--batchnorm', default=False, type=bool,
                    help="Add batch normalization to each layer in the adv models")
parser.add_argument('-nns', '--nn_size', default=64, type=int,
                    help="size of each nn layer")
parser.add_argument('-nnl', '--layers', default=8, type=int,

                    help="amount of layers")

parser.add_argument('-nne', '--nn_epochs', default=20, type=int,help="Number of epochs for the NN")

args = parser.parse_args()
SEPARATOR = '\\' if platform == 'win32' else '/'


def get_datasets():
    config = json.load(open("dataset_config.json"))["datasets"]
    return (config.keys())


def normalize(dataset, verbose=False):
    norm = Normalizator(dataset=dataset)
    if args.method == "all":
        method = ["zscore", "minmax", "tanh", "variablescaling"]
        for m in method:
            print("Running normalization method: {}".format(m))
            norm.normalize(method=m, save=True)

    else:
        norm.normalize(args.method, save=True)
    if verbose:
        print(norm.df.describe())
        print(norm.df_norm.describe())


def main():
    """
    Here we run the pipeline
        we can run with all datasets (not implemeted yet) or with a specific one
    """
    datasets = get_datasets()
    if args.dataset == "all":
        output_basic = {}
        output_advanced = {}

        for dataset in datasets:
            print("Running dataset: {}".format(dataset))
            if "compHardware" in dataset:
                print("skipping comp hardware")
                continue
            normalize(dataset, verbose=False)
            if args.model in ["all", "basic"]:
                print('----BASIC models----')
                output_dataset_basic = run_basic_models(args, dataset)
                output_basic = {**output_basic, **output_dataset_basic}

            if args.model in ["all", "adv"]:
                print('----ADV models----')
                output_dataset_advanced = run_advanced_models(args, dataset)
                output_advanced = {**output_advanced, **output_dataset_advanced}

    else:  # run for a single one
        print("SINGLE DATA")
        if not args.dataset in datasets:
            print("Dataset not found")
            sys.exit(1)

        output_basic = {}
        output_advanced = {}
        dataset = args.dataset
        normalize(dataset, verbose=False)
        output_dataset_basic = run_basic_models(args, dataset)
        print(output_dataset_basic)

        output_basic = {**output_basic, **output_dataset_basic}

        output_dataset_advanced = run_advanced_models(args, dataset)
        print('--------')
        print(output_dataset_advanced)
        output_advanced = {**output_advanced, **output_dataset_advanced}

        # ! NORMALIZATION - MAT
        # * Write the input and output of your normalization method here
        # * Output results in output/post_norma_data with proper names (e.g wine_zscore.csv)
        normalize(args.dataset)

        # ! BASIC MODELS - JOHAN
        # * Reads normalized data from output/post_norma_data in the specified format and runs classifiers
        # * Reads the unnormalized data from datasets

        output_basic = run_basic_models(args, dataset)
        output_advanced = run_advanced_models(args, dataset)



    
    #! FIX FILE PATH
    if args.model in ["all", "basic"]:
        with open('output' + SEPARATOR + 'results' + SEPARATOR + "predictions" + SEPARATOR + start_time + "_basic" + '.pkl', 'wb') as fp:
            pickle.dump(output_basic, fp)

    if args.model in ["all", "adv"]:        

        name = f"_adv({args.nn_epochs},{args.layers},{args.nn_size},"
        if args.batchnorm:
            name+="BN"
        
        
        name+=")"

        with open('output' + SEPARATOR + 'results' + SEPARATOR + "predictions" + SEPARATOR + name + str(time.time()) + '.pkl', 'wb') as fp:
            pickle.dump(output_advanced, fp)

    # ! EVALUATION - ISAK
    # Write the input you want to evaluate here and the output you will produce


if __name__ == "__main__":
    now = datetime.now()
    start_time = now.strftime("%d%m%Y %H%M%S")
    main()
