# Python file to read the results of the experiments and generate the output tables, plots and graphs.
import pickle
from sys import platform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


SEPARATOR = '\\' if platform == 'win32' else '/'

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            print(key)
            recursive_items(value)
        else:
            print(key)

def evaluate(output):
    for ds in output:
        type = output[ds]['pred_type']

        for model in output[ds]:
            if model == 'pred_type':
                continue
            y_test, y_pred = output[ds][model]['true'], output[ds][model]['pred']
            if type == "regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print("--------------------------------------")
                print(ds + ' ' + model)
                print('MAE is {}'.format(mae))
                print('MSE is {}'.format(mse))
                print('R2 score is {}'.format(r2))
            else:
                print("--------------------------------------")
                print(ds + ' ' + model)
                print("Classification report")
                print(classification_report(y_test, y_pred))
                print("Confusion matrix")
                print(confusion_matrix(y_test, y_pred))
                print("Accuracy score")
                print(accuracy_score(y_test, y_pred))

file = "output/results/28092021 225417.pkl"
a_file = open(file, "rb")
output = pickle.load(a_file)
evaluate(output)
