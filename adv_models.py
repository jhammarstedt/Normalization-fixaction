from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import json

from sys import platform
from normal_methods import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

SEPARATOR = '\\' if platform == 'win32' else '/'

class ModelClass():
    def __init__(self,data:dict, NN_layers=3, NN_size=32, epochs=50, batch_size=10) -> None:
        self.datasets = data
        self.layers = NN_layers
        self.layer_size = NN_size
        self.epochs = epochs
        self.batch_size = batch_size

    def run_models(self):
        """#!ISAK From this method you can return whatever you want to get to your output """
        
        for dataset_name in self.datasets.keys():
            print(f"############# DATASET NAME AND METHOD: {dataset_name} ############")
            df = self.datasets[dataset_name]["data"].copy() #copy dataframe 
            
            categorical = df.select_dtypes('category')
            
            df[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            target = self.datasets[dataset_name]["target"]
            

            if self.datasets[dataset_name]["pred_type"] =="regression":
                ### TODO: adapt model
                print("regression")
            elif self.datasets[dataset_name]["pred_type"]=="classification": 
                ### TODO: adapt model
                print("classification")
            else: #both, run all
                raise TypeError("Prediction type not supported")
      
            X = df.drop([target], axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            y_pred = self.train_model(X_train,X_test,y_train,y_test, self.datasets[dataset_name]["pred_type"])    
            
    def evaluate(self,y_test,y_pred, type="regression"):
        if type == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("The model performance for testing set")
            print("--------------------------------------")
            print('MAE is {}'.format(mae))
            print('MSE is {}'.format(mse))
            print('R2 score is {}'.format(r2))
        else:
            print("Classification report")
            print(classification_report(y_test, y_pred))
            print("Confusion matrix")
            print(confusion_matrix(y_test, y_pred))
            print("Accuracy score")
            print(accuracy_score(y_test, y_pred))

    def train_model(self,X_train,X_test,y_train,y_test, pred_type="classification"):
        """
        Here we train the models and get access to data for evaluation
        
        """
        print("**********NN************")
        model = Sequential()
        
        size = self.layer_size
        model.add(Dense(size, input_dim=X_train.shape[1], activation='relu'))
        for i in range(1, self.layers - 1):
            size = int(size / 2)
            model.add(Dense(size, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        
        _, accuracy = model.evaluate(X_train, y_train)
        print('Accuracy: %.2f' % (accuracy*100))
        
        y_pred =  model.predict(X_test)
        # print(y_test)
        # print(y_pred)
        if (pred_type == "classification"):
            y_pred = np.where( np.array(y_pred) > 0.5, 1, 0)
        self.evaluate(y_test,y_pred,type=pred_type)
        print("************************")

        return y_pred

def read_data(dataset_name):
    path = os.path.join(os.getcwd(),rf"output{SEPARATOR}post_norma_data")
    files = glob.glob(os.path.join(path,f"{dataset_name}*.csv"))
    if len(files)==0:
        return None #no data here
    
    #files.append(os.path.join(path,f"{dataset_name}_.csv"))

    config = json.load(open("dataset_config.json"))["datasets"]
    datasets = {}
    for f in files:
        
        norm_method = os.path.basename(f)
        print(f"Training model for {norm_method.split('_')[1].upper()}")
        
        dtype = config[dataset_name]["dtype"]
        df = pd.read_csv(f,dtype=dtype)
        datasets[norm_method] = {"data":df,"target":config[dataset_name]["target"],"pred_type":config[dataset_name]["pred_type"],"pred_type":config[dataset_name]["pred_type"]}

    #adding unnormalized data for comparison
    datasets[f"{dataset_name}_UnNorm"] = {"data":load_data(dataset_name),"target":config[dataset_name]["target"],"pred_type":config[dataset_name]["pred_type"]}
    return datasets
    
def run_advanced_models(dataset)->str:
    data = read_data(dataset)
    if data is None:
        return "No data available for dataset"
    else:
        print("Running advanced models for dataset {}".format(dataset))
        models = ModelClass(data, )
        models.run_models()
        return "Models trained"