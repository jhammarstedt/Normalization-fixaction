# Normalization-fixaction
<img src ="https://miro.medium.com/max/271/0*d5_CPfpuJ2uIjIk3" align= "right">
This project in collaboration with Spotify, focusing on the effect of normalization, including standardizing, scaling and more, on the training and inference of machine learning models. A lot of focus in machine learning is on models and less so on data and how to treat it. In this project we will focus on the impact of data on model training.

## Authors
* Johan Hammarstedt - [jhammarstedt](https://github.com/jhammarstedt)
* Isak Persson - [isakper](https://github.com/isakper)
* Matej Sestak - [sestys](https://github.com/sestys)
* João Ferreira - [JoaoFerreira100](https://github.com/JoaoFerreira100)

## Setup

```pip install -r requirements.txt ```

Use the dataset_config to write info and call the dataset

Once everything is done we run it all from the run_pipeline.py script 

### Structure:
0. Check the datasets_config and add all relevant info to your dataset here & the same if we add more in the future
1. Normal_methods.py : Data is normalized and saved to post_norm_data as CSV files
2. basic_models.py and adv_models.py:  Ouput in output/post_norm_data is read and models are trained
    * Basic are saved to output/training/models/basic
    * Networks are saved to output/training/models/networks
    * model output (accuracy, training info etc) are saved to: output/training/training_results/{model_name}_{score_type}_{timestamp}
3. gather_results.py: Results are read in appropriate format, cleaned, compared and displayed in proper format, 



## Run
To run the pipeline choose dataset and method with flags -d and -m
`python run_pipeline.py -d wine -m zscore`

To run for everything:
`python run_pipeline.py -d wine -m all

Note that running for all datasets is not working yet since all have yet to be added.

### Parameters
#### Datasets: 
* wine
* compHardware
* adult
* breastCancer
* mechanicalHardware

#### methods:
* zscore
* tanh
* minmax
* variablescaling

#### Models:
* Networks (Regression with ReLU and Classification with Sigmoid) - Testing with and without Batch norm
* SVM and SVR
* KNN
