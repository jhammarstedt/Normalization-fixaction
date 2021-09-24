## Training pipeline for KNN, logistic regression and SVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


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

if __name__ == "__main__":
    run_model()