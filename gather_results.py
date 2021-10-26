# Python file to read the results of the experiments and generate the output tables, plots and graphs.
import pickle
from sys import platform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import collections
import csv
import plotly.express as px
import numpy as np
import pandas as pd

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
    loss_res = collections.defaultdict(
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

                if 'train_loss' in output[ds][model].keys():
                    loss_res[ds_name]['pred_type'] = type
                    loss_res[ds_name][model][norm_type]['train_loss'].append(output[ds][model]['train_loss'])
                    loss_res[ds_name][model][norm_type]['val_loss'].append(output[ds][model]['val_loss'])
                    if type == 'classification':
                        loss_res[ds_name][model][norm_type]['train_acc'].append(output[ds][model]['train_acc'])
                        loss_res[ds_name][model][norm_type]['val_acc'].append(output[ds][model]['val_acc'])

    # results for plotting
    for ds in loss_res:

        pred_type = loss_res[ds]['pred_type']

        for model in loss_res[ds]:
            if model == 'pred_type':
                continue

            df_train_loss = pd.DataFrame()
            df_val_loss = pd.DataFrame()

            df_train_acc = pd.DataFrame()
            df_val_acc = pd.DataFrame()

            for norm_type in loss_res[ds][model]:
                train_loss = np.mean(loss_res[ds][model][norm_type]['train_loss'], axis=0)
                val_loss = np.mean(loss_res[ds][model][norm_type]['val_loss'], axis=0)

                df_train_loss[model + norm_type] = train_loss
                df_val_loss[model + norm_type] = val_loss

                if pred_type == 'classification':
                    train_acc = np.mean(loss_res[ds][model][norm_type]['train_acc'], axis=0)
                    val_acc = np.mean(loss_res[ds][model][norm_type]['val_acc'], axis=0)

                    df_train_acc[model + norm_type] = train_acc
                    df_val_acc[model + norm_type] = val_acc

            fig = px.line(df_train_loss)
            fig.update_layout(title_text=ds + ' train loss')
            fig.show()

            fig = px.line(df_val_loss)
            fig.update_layout(title_text=ds + ' val loss')
            fig.show()

            if pred_type == 'classification':
                fig = px.line(df_train_acc)
                fig.update_layout(title_text=ds + ' train acc')
                fig.show()

                fig = px.line(df_val_acc)
                fig.update_layout(title_text=ds + ' val acc')
                fig.show()


    # results for csv
    for ds in res:
        with open(output_dir + SEPARATOR + ds + '.csv', 'w') as f:
            writer = csv.writer(f)

            # get the header
            header = ['Dataset', 'Norm type', 'Metric'] + [model for model_ in
                                                           list(list(res[ds].values())[0].values())[0] for model in
                                                           [model_ + '_mean'] + [model_ + '_std']]
            writer.writerow(header)
            # results for csv
            for norm_type in res[ds]:
                for metric in res[ds][norm_type]:
                    row = [ds, norm_type, metric]
                    for model in res[ds][norm_type][metric]:
                        scaler = StandardScaler()
                        scaler.fit(res[ds][norm_type][metric][model])
                        row.extend([scaler.mean_[0], scaler.var_[0]**0.5])

                    writer.writerow(row)


file_names = ["26102021 174253_advanced.pkl"]
file_paths = ["output" + SEPARATOR + "results" + SEPARATOR + "predictions" + SEPARATOR + f for f in file_names]
evaluate(file_paths, "output/results/final_results")
