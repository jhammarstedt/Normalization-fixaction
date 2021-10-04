# Python file to read the results of the experiments and generate the output tables, plots and graphs.
import pickle
from sys import platform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import collections
import csv

SEPARATOR = '\\' if platform == 'win32' else '/'


def makehash():
    return collections.defaultdict(makehash)


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            print(key)
            recursive_items(value)
        else:
            print(key)


def evaluate(files, output_dir):
    '''
    res = {'ds': {'metric': {'model': ...}}}
    '''
    res = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
    for file in files:
        output = open(file, "rb")
        output = pickle.load(output)

        for ds in output:
            type = output[ds]['pred_type']

            for model in output[ds]:
                if model == 'pred_type':
                    continue
                y_test, y_pred = output[ds][model]['true'], output[ds][model]['pred']

                ds_name, norm_type = ds.split('_')
                norm_type = norm_type[:-4]
                if type == "regression":
                    res[ds_name][norm_type]['MAE'][model].append([mean_absolute_error(y_test, y_pred)])
                    res[ds_name][norm_type]['MSE'][model].append([mean_squared_error(y_test, y_pred)])
                    res[ds_name][norm_type]['R2'][model].append([r2_score(y_test, y_pred)])
                else:
                    res[ds_name][norm_type]['ACC'][model].append([accuracy_score(y_test, y_pred)])
                    res[ds_name][norm_type]['AUC'][model].append([roc_auc_score(y_test, y_pred)])

    for ds in res:
        with open(output_dir + SEPARATOR + ds + '.csv', 'w') as f:
            writer = csv.writer(f)

            # get the header
            header = ['Dataset', 'Norm type', 'Metric'] + [model for model_ in
                                                           list(list(res[ds].values())[0].values())[0] for model in
                                                           [model_ + '_mean'] + [model_ + '_var']]
            writer.writerow(header)

            for norm_type in res[ds]:
                for metric in res[ds][norm_type]:
                    row = [ds, norm_type, metric]
                    for model in res[ds][norm_type][metric]:
                        scaler = StandardScaler()
                        scaler.fit(res[ds][norm_type][metric][model])
                        row.extend([scaler.mean_[0], scaler.var_[0]])

                    writer.writerow(row)


file_names = ["04102021 153511_basic.pkl", "04102021 162329_basic.pkl", "04102021 163002_basic.pkl",
         "04102021 163455_basic.pkl", "04102021 164031_basic.pkl"]
file_paths = ["output/results/" + f for f in file_names]
evaluate(file_paths, "output/results/")
