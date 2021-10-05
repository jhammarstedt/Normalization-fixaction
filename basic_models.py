# Training pipeline for KNN, logistic regression and SVM
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier


from helpers import evaluate_results, read_data


class ModelClass:
    def __init__(self, data: dict, seed=0) -> None:
        self.datasets = data
        self.seed = seed

    def run_models(self):
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
                y_pred = train_model(X_train, X_test, y_train, y_test, model=model)
                results[dataset_name][model] = {'pred': list(y_pred), 'true': list(y_test)}

        return results


def train_model(X_train, X_test, y_train, y_test, model=None):
    """
    Here we train the models and get access to data for evaluation

    """
    if not model:
        raise TypeError("Need to specify model type")
    elif model == "knn":
        print("**********KNN************")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("************************")

    elif model == "svm":
        # SVM
        print("**********SVM************")
        svm = SVC()
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("************************")
    elif model == "logreg":
        # Logistic Regression
        print("**********Logistic Regression************")
        logreg = LogisticRegression()  # solver='lbfgs', max_iter=100)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="classification")
        print("***************************")
    elif model == "XGBR":
        # XGBoost Regressor
        print("**********XGBoost Regressor************")
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        evaluate_results(y_test, y_pred, model_type="regression")
        print("***************************")
    elif model == "svr":
        # SVR
        print("**********SVR************")
        svr = SVR()
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

# if __name__ == "__main__":
#     data = read_data()
#     model = ModelClass(data)
#     model.run_models()
