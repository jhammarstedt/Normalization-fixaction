# Python script to run the pipeline
import sys
import argparse
parser = argparse.ArgumentParser()
import normal_methods,basic_models
from normal_methods import Normalizator


#parser.add_argument("-a","--all",default = "all_data",help="Run with all datasets")
#args = parser.parse_args()

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
