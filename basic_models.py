# Training pipeline for KNN, logistic regression and SVM
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from pprint import pprint


from helpers import evaluate_results, read_data


class ModelClass:
    def __init__(self, data: dict, seed=0) -> None:
        self.datasets = data
        self.seed = seed

    def run_models(self, grid_search=False):
        """#!ISAK From this method you can return whatever you want to get to your output """

        results = {}
        for dataset_name in self.datasets.keys():

            print(f"#############DATASET NAME AND METHOD: {dataset_name} ############")
            df = self.datasets[dataset_name]["data"].copy()

            categorical = df.select_dtypes('category')

            df[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            target = self.datasets[dataset_name]["target"]

            if self.datasets[dataset_name]["pred_type"] == "regression":
                model_class = ["svr", "XGBR"]
            elif self.datasets[dataset_name]["pred_type"] == "classification":
                model_class = ["svm", "knn", "logreg"]
            else:  # both, run all
                raise TypeError("Prediction type not supported")
                # model_class = ["svm","knn","logreg","XGBR","svr"]

            X = df.drop([target], axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

            results[dataset_name] = {"pred_type": self.datasets[dataset_name]["pred_type"]}
            for model in model_class:
                # Runs the model and returns the predictions on the test set
                y_pred = train_model(X_train, X_test, y_train, y_test, model_name=model, grid_search=grid_search)
                results[dataset_name][model] = {'pred': list(y_pred), 'true': list(y_test)}

        return results


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
                'n_neighbors': [3, 5],
                'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree']
            }
        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(knn, grid_search_params, n_jobs=4)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            knn.set_params(**clf.best_params_)

        knn.fit(X_train.values, y_train)
        y_pred = knn.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("************************")

    elif model_name == "svm":
        print("**********SVM************")
        svm = SVC(**params)

        if grid_search_params is None:
            grid_search_params = {
                'C': [1, 2],
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2, 3]
            }

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(svm, grid_search_params, n_jobs=4)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
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
                'C': [1, 2],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            }

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(logreg, grid_search_params, n_jobs=4)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
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
            clf = GridSearchCV(xgb, grid_search_params, n_jobs=4)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            xgb.set_params(**clf.best_params_)

        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="regression")
        print("***************************")

    elif model_name == "svr":
        print("**********SVR************")
        svr = SVR(**params)
        if grid_search_params is None:
            grid_search_params = {
                'C': [1, 2],
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2, 3]
            }

        if grid_search and grid_search_params:
            print("Performing grid search over supplied parameteres")
            pprint(grid_search_params)
            clf = GridSearchCV(svr, grid_search_params, n_jobs=4)
            clf.fit(X_train.values, y_train)
            print('Best parameters found:')
            pprint(clf.best_params_)
            svr.set_params(**clf.best_params_)

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="regression")
        print("***************************")
    else:
        raise TypeError("Model type not supported")

    return y_pred


def run_basic_models(args, dataset):
    """Function that runs the model training"""
    data = read_data(dataset)
    if data is None:
        return "No data available for dataset"
    else:
        print("Running basic models for dataset {}".format(dataset))
        models = ModelClass(data, args.seed)
        results = models.run_models()
        return results


if __name__ == "__main__":
    data = read_data('adult')
    model = ModelClass(data)
    model.run_models(grid_search=True)
