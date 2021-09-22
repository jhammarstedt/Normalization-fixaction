import os
from sys import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import math
from sklearn.preprocessing import MinMaxScaler

try: 
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError: 
    pass

def z_score(data):
    data_copy = data.copy()
    scaler = StandardScaler()
    data_without_cat = data_copy.select_dtypes(exclude='category')
    data_copy[data_without_cat.columns] = scaler.fit_transform(data_without_cat)
    return data_copy
        
def tanh_norm(df):
    #Reduce influence of the values in the tail of the distribution

    m = np.mean(df.iloc[:,:-1], axis=0) # array([16.25, 26.25])
    std = np.std(df.iloc[:,:-1], axis=0) # array([17.45530005, 22.18529919])

    tanh_df = 0.5 * (np.tanh(0.01 * ((df.iloc[:,:-1] - m) / std)) + 1)
    return tanh_df

def min_max(data):
    scaler = MinMaxScaler()
    data.iloc[:,:-1] = scaler.fit_transform(data.iloc[:,:-1])
    return data

def pareto_scaling(df):
    data_copy = df.copy()
    data_without_cat = data_copy.select_dtypes(exclude='category')
    mean = np.mean(data_without_cat, axis=0)
    std = np.std(data_without_cat, axis=0)
    data_copy[data_without_cat.columns] = (data_without_cat - mean) / np.sqrt(std)
    return data_copy

def variable_stability_scaling():
    pass

def load_data(dataset:str):
    cwd = os.getcwd()
    separator = '\\' if platform == 'win32' else '/'
    if dataset == "wine":
        w = pd.read_csv(r"C:\Users\johan\Documents\GitHub\kth_private\project_course\datasets\wine\winequality-white.csv",delimiter=";")
        r  = pd.read_csv(r"C:\Users\johan\Documents\GitHub\kth_private\project_course\datasets\wine\winequality-red.csv",delimiter=";")
        r['color'] = 'red'
        w['color'] = 'white'
        df = r.append(w)
        df = df.sample(frac=1).reset_index(drop=True)
    elif dataset == "adult":
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        dtypes = {'age': np.int64, 'workclass': 'category', 'fnlwgt': np.int64, 'education': 'category', 'education_num': np.int64, 
                  'marital_status': 'category', 'occupation': 'category', 'relationship': 'category', 'race': 'category', 'sex': 'category', 
                  'capital_gain': np.int64, 'capital_loss': np.int64, 'hours_per_week': np.int64, 'native_country': 'category', 'income': 'category'}
        df = pd.read_csv(cwd + separator + 'datasets' + separator + 'adult' + separator + 'adult.data', delimiter=', ', names=column_names, dtype=dtypes, index_col=False)
    elif dataset == "comp":
        df = pd.read_csv(
            cwd + separator + 'datasets' + separator + 'comp' + separator + 'machine.data',
            delimiter=",",
            header=None)
        df = df.iloc[:,2:]
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        raise NameError("Not implemented yet")
    return df

class Normalizator():
    def __init__(self, dataset:str):
        """
        Class that reads data and performs normalization methods
        Args:
            - dataset : str of the type of dataset to use, not that this also needs to be implemented as a separete field in load_data
        Methods:
            - normalize: function to normalize the data given method
            - boxplot: creates subplots of all variables
            
        """
        
        self.dataset = dataset
        self.df = load_data(self.dataset)
        self.df_norm = None
        
    

    def normalize(self,method="zscore",reset=False):
        # just to make sure the data is not tampered with before normalization
        if reset:
            self.df = load_data(self.dataset) 
        if method == "zscore": 
            self.df_norm = z_score(self.df)
        elif method == "tanh":
            self.df_norm = tanh_norm(self.df) 
        elif method == "pareto":
            print("pareto scaling")
            self.df_norm = pareto_scaling(self.df)
        else:
            raise NotImplementedError("This normalization is not implemented yet")
        
    def visuals(self):
        fig, axes = plt.subplots(3,4,figsize=(15,15))
        for i,el in enumerate(list(self.df.columns.values)[:-1]):
                a = self.df.boxplot(el, ax=axes.flatten()[i])

        
        b = self.df.hist(figsize=(15,15))
        plt.tight_layout() 

        plt.show()

    def lblEncoder(self,col):
        le = LabelEncoder()
        le.fit(self.df[col])
        self.df[col] = le.transform(self.df[col])
        return self.df

    def run_model(self,model=None):
        if not model:
            raise TypeError("Need to specify model type")
        elif model =="knn":
            categorical = self.df_norm.select_dtypes('category')
            self.df_norm[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            X = self.df_norm.iloc[:,:-1]
            y = self.df_norm.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))


if __name__== "__main__":
    n = Normalizator("wine")
    n.normalize(method="zscore")
    n.visuals()
    n.run_model(model="knn")
