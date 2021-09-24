# Python script to run the pipeline
import sys
import argparse
parser = argparse.ArgumentParser()
import normal_methods,basic_models

parser.add_argument("-a","--all",default = "all_data",help="Run with all datasets")
args = parser.parse_args()

if __name__ == "__main__":
    data_file = sys.argv[1]
    print(data_file)
    print(args.all)
    #normal_methods.run_normal_methods(data_file)
    #basic_models.run_basic_models(data_file)
