import os
import glob
import json
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sys import platform

SEPARATOR = '\\' if platform == 'win32' else '/'


def evaluate_results(y_test, y_pred, model_type="regression"):
    if model_type == "regression":
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


def load_data(dataset: str, get_config=False):
    cwd = os.getcwd()
    config = json.load(open("dataset_config.json"))["datasets"]
    if dataset == "wine":
        dtypes = config["wine"]["dtype"]

        df = pd.read_csv(rf"datasets{SEPARATOR}wine{SEPARATOR}wine.csv", delimiter=",", dtype=dtypes, engine='python')

    elif dataset == "adult":
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                        'relationship',
                        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        dtypes = config["adult"]["dtype"]
        df = pd.read_csv(cwd + SEPARATOR + 'datasets' + SEPARATOR + 'adult' + SEPARATOR + 'adult.data', delimiter=', ',
                         names=column_names, dtype=dtypes, index_col=False, engine='python')
    elif dataset == "compHardware":
        dtypes = config["compHardware"]["dtype"]

        df = pd.read_csv(
            cwd + SEPARATOR + 'datasets' + SEPARATOR + 'compHardware' + SEPARATOR + 'machine.data',
            delimiter=",", dtype=dtypes, index_col=False, engine='python')
        #df = df.iloc[:, 2:]
        #df = df.sample(frac=1).reset_index(drop=True)

    elif dataset == "breastCancer":
        dtypes = config["breastCancer"]["dtype"]
        column_names = ["Diagnosis", "radius_1", "texture_1", "perimeter_1", "area_1", "smoothness_1",
                        "compactness_1", "concavity_1", "concave_points_1", "symmetry_1", "fractal_dimension_1",
                        "radius_2", "texture_2", "perimeter_2", "area_2", "smoothness_2", "compactness_2",
                        "concavity_2", "concave_points_2", "symmetry_2", "fractal_dimension_2", "radius_3", "texture_3",
                        "perimeter_3", "area_3", "smoothness_3", "compactness_3", "concavity_3", "concave_points_3",
                        "symmetry_3", "fractal_dimension_3"]

        df = pd.read_csv(rf"datasets{SEPARATOR}breastCancer{SEPARATOR}wdbc.data", delimiter=",", names=column_names,
                         dtype=dtypes, engine='python')

    elif dataset == "tempForecast":
        dtypes = config["tempForecast"]["dtype"]


        df = pd.read_csv(rf"datasets{SEPARATOR}tempForecast{SEPARATOR}Bias_correction_ucl.csv", delimiter=",",
                         dtype=dtypes, engine='python')

    elif dataset == "CCPP":
        dtypes = config["CCPP"]["dtype"]


        df = pd.read_csv(rf"datasets{SEPARATOR}CCPP{SEPARATOR}Folds5x2_pp.csv", delimiter=",",
                         dtype=dtypes, engine='python')

    else:
        raise NameError("Not implemented yet")
    if get_config:
        return df, config[dataset]
    else:
        return df


def read_data(dataset_name):
    path = os.path.join(os.getcwd(), f"output{SEPARATOR}post_norma_data")
    files = glob.glob(os.path.join(path, f"{dataset_name}*.csv"))
    if len(files) == 0:
        return None  # no data here

    # files.append(os.path.join(path,f"{dataset_name}_.csv"))

    config = json.load(open("dataset_config.json"))["datasets"]
    datasets = {}
    for f in files:
        norm_method = os.path.basename(f)
        print(f"Training model for {norm_method.split('_')[1].upper()}")

        dtype = config[dataset_name]["dtype"]
        df = pd.read_csv(f, dtype=dtype)
        datasets[norm_method] = {
            "data": df,
            "target": config[dataset_name]["target"],
            "pred_type": config[dataset_name]["pred_type"]
        }

    # adding unnormalized data for comparison
    datasets[f"{dataset_name}_UnNorm"] = {"data": load_data(dataset_name), "target": config[dataset_name]["target"],
                                          "pred_type": config[dataset_name]["pred_type"]}
    return datasets
