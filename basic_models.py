## Training pipeline for KNN, logistic regression and SVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from normal_methods import load_data
from sys import platform


SEPARATOR = '\\' if platform == 'win32' else '/'
class ModelClass():
    def __init__(self,data:dict) -> None:
        self.datasets = data
        


    def run_models(self):
         
        #model_class =["knn","logreg","svm"]
        for dataset_name in self.datasets.keys():
            
            df = self.datasets[dataset_name]["data"].copy() #copy dataframe 
            
            categorical = df.select_dtypes('category')
            
            df[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            target = self.datasets[dataset_name]["target"]
            

            if self.datasets[dataset_name]["pred_type"] =="regression":
                model_class = ["svr","XGBR"]
            elif self.datasets[dataset_name]["pred_type"]=="classification": 
                model_class = ["svm","knn","logreg"]
            else: #both, run all
                raise TypeError("Prediction type not supported")
                #model_class = ["svm","knn","logreg","XGBR","svr"]
      
            X = df.drop([target], axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for model in model_class:
                self.train_model(X_train,X_test,y_train,y_test,model=model)            
            
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

               
    def train_model(self,X_train,X_test,y_train,y_test,model=None):
        if not model:
            raise TypeError("Need to specify model type")
        elif model =="knn":
            print("**********KNN************")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            self.evaluate(y_test,y_pred,type="classification")
            print("************************")

        elif model == "svm":
            # SVM
            print("**********SVM************") 
            svm = SVC()
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            self.evaluate(y_test,y_pred,type="classification")
            print("************************")
        elif model=="logreg":
            # Logistic Regression
            print("**********Logistic Regression************")
            logreg = LogisticRegression()#solver='lbfgs', max_iter=100)
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            self.evaluate(y_test,y_pred,type="classification")
            print("***************************")
        elif model == "XGBR":
            # XGBoost Regressor
            print("**********XGBoost Regressor************")
            xgb = XGBRegressor()
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            self.evaluate(y_test,y_pred,type="regression")
            print("***************************")
        elif model == "svr":
            # SVR
            print("**********SVR************")
            svr = SVR()
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)
            self.evaluate(y_test,y_pred,type="regression")
            print("***************************")
        else:
            raise TypeError("Model type not supported")

def read_data(dataset_name):
    path = os.path.join(os.getcwd(),"output\post_norma_data")
    files = glob.glob(os.path.join(path,f"{dataset_name}*.csv"))
    if len(files)==0:
        return None #no data here
    
    #files.append(os.path.join(path,f"{dataset_name}_.csv"))

    config = json.load(open("dataset_config.json"))["datasets"]
    datasets = {}
    for f in files:
        #dataset_name = (os.path.basename(f).split('_'))[0]
        dtype = config[dataset_name]["dtype"]
        df = pd.read_csv(f,dtype=dtype)
        datasets[dataset_name] = {"data":df,"unormalized":load_data(dataset_name),"target":config[dataset_name]["target"],"pred_type":config[dataset_name]["pred_type"],"pred_type":config[dataset_name]["pred_type"]}

        
    return datasets

def run_basic_models(dataset)->str:
    """Function that runs the model training"""
    data = read_data(dataset)
    if data is None:
        return "No data available for dataset"
    else:
        print("Running basic models for dataset {}".format(dataset))
        models = ModelClass(data)
        models.run_models()
        return "Models trained"
# if __name__ == "__main__":
#     data = read_data()
#     model = ModelClass(data)
#     model.run_models()