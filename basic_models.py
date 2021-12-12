# Training pipeline for KNN, logistic regression and SVM
import numpy as np
import warnings

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from pprint import pprint


from helpers import evaluate_results, read_data, save_json, load_json


warnings.simplefilter("ignore", category=ConvergenceWarning)

class ModelClass:
    def __init__(self, data: dict, seed=0) -> None:
        self.datasets = data
        self.seed = seed

    def run_models(self, params=None, grid_search=False):
        """#!ISAK From this method you can return whatever you want to get to your output """

        results = {}
        best_params = {}
        for dataset_name in self.datasets.keys():

            print(f"#############DATASET NAME AND METHOD: {dataset_name} ############")
            df = self.datasets[dataset_name]["data"].copy()

            categorical = df.select_dtypes('category')

            df[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            target = self.datasets[dataset_name]["target"]

            if self.datasets[dataset_name]["pred_type"] == "regression":
                model_class = ["svr", "XGBR"]
                # model_class = []
            elif self.datasets[dataset_name]["pred_type"] == "classification":
                # model_class = ["svm", "knn", "logreg"]
                # model_class = ['svm', "knn"]
                model_class = []
                continue

            else:  # both, run all
                raise TypeError("Prediction type not supported")
                # model_class = ["svm","knn","logreg","XGBR","svr"]

            X = df.drop([target], axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

            results[dataset_name] = {"pred_type": self.datasets[dataset_name]["pred_type"]}
            best_params[dataset_name] = {}
            for model in model_class:
                # Runs the model and returns the predictions on the test set
                model_params = {}
                if not grid_search and params[dataset_name][model]:
                    model_params = params[dataset_name][model]
                y_pred, best_model_params = train_model(X_train, X_test, y_train, y_test, params=model_params, model_name=model, grid_search=grid_search)
                results[dataset_name][model] = {'pred': list(y_pred), 'true': list(y_test)}
                best_params[dataset_name][model] = best_model_params

        return results, best_params


def train_model(X_train, X_test, y_train, y_test, model_name=None, params=None, grid_search=False, grid_search_params=None):
    """
    Here we train the models and get access to data for evaluation

    """
    if params is None:
        params = {}
    if not model_name:
        raise TypeError("Need to specify model type")
    elif model_name == "knn":
        print("**********KNN************")
        if not params:
            params['n_neighbors'] = 3

        knn = KNeighborsClassifier(**params)
        if grid_search_params is None:
            grid_search_params = {
                'n_neighbors': [3, 5, 10, 20, 50, 100],
                'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree']
            }
        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(knn, grid_search_params, n_jobs=4, cv=3)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            params = clf.best_params_
            knn.set_params(**clf.best_params_)

        knn.fit(X_train.values, y_train)
        y_pred = knn.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("************************")

    elif model_name == "svm":
        print("**********SVM************")
        svm = SVC(**params, max_iter=30000)

        if grid_search_params is None:
            grid_search_params = [
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['rbf'],
                    'gamma': [0.1, 0.001, 0.0001],
                    # 'degree': [2, 3]
                },
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['poly'],
                    'gamma': [1, 0.1, 0.001, 0.0001],
                    'degree': [2, 3]
                },
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['linear']
                }
            ]

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(svm, grid_search_params, n_jobs=4, cv=3)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            params = clf.best_params_
            svm.set_params(**clf.best_params_)

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("************************")
    elif model_name == "logreg":
        print("**********Logistic Regression************")
        logreg = LogisticRegression(**params)  # solver='lbfgs', max_iter=100)

        if grid_search_params is None:
            grid_search_params = {
                "C": np.logspace(-3,3,7),
                "penalty": ["l1", "l2"]
            }

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(logreg, grid_search_params, n_jobs=4, cv=3)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            params = clf.best_params_
            logreg.set_params(**clf.best_params_)

        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("***************************")
    elif model_name == "XGBR":
        print("**********XGBoost Regressor************")
        xgb = XGBRegressor(**params)

        if grid_search_params is None:
            grid_search_params = {
                'objective': ['reg:linear'],
                'learning_rate': [.03, 0.05, .07],
                'max_depth': [5, 6, 7],
                'min_child_weight': [4],
                'silent': [1],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'n_estimators': [100]
            }

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(xgb, grid_search_params, n_jobs=4, cv=3)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            params = clf.best_params_
            xgb.set_params(**clf.best_params_)

        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="regression")
        print("***************************")

    elif model_name == "svr":
        print("**********SVR************")
        svr = SVR(**params)
        if grid_search_params is None:
            grid_search_params = [
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['rbf'],
                    'gamma': [0.1, 0.001, 0.0001],
                    # 'degree': [2, 3]
                },
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['poly'],
                    'gamma': [1, 0.1, 0.001, 0.0001],
                    'degree': [2, 3]
                },
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['linear']
                }
            ]

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(svr, grid_search_params, n_jobs=4, cv=3)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            params = clf.best_params_
            svr.set_params(**clf.best_params_)

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="regression")
        print("***************************")
    else:
        raise TypeError("Model type not supported")

    return y_pred, params


def run_basic_models(args, dataset, gridsearch=False):
    """Function that runs the model training"""
    data = read_data(dataset)
    if data is None:
        return "No data available for dataset"
    else:
        params_path = 'output/results/gridsearch/params2_{}.json'.format(dataset)
        print("Running basic models for dataset {}".format(dataset))
        models = ModelClass(data, args.seed)
        params = {}
        if not gridsearch:
            params = load_json(params_path)
        results, params = models.run_models(params=params, grid_search=gridsearch)

        if not gridsearch:
            save_json(params, params_path)
        return results


if __name__ == "__main__":
    data = read_data('wine')
    model = ModelClass(data)
    result, best_params = model.run_models(grid_search=True)
    pprint(best_params)
    # pprint(result)
    save_json(best_params, 'output/results/gridsearch/params_wine2.json')
    # save_json(result, 'output/results/final_results/bank.json')


