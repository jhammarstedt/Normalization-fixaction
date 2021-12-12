# Python file to read the results of the experiments and generate the output tables, plots and graphs.
import pickle
from sys import platform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from dash import dash_table

import os
import collections
import csv
import plotly.express as px
import numpy as np
import pandas as pd
import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
pd.options.display.float_format = '${:.2f}'.format
SEPARATOR = '\\' if platform == 'win32' else '/'


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def get_datasets():
    config = json.load(open("dataset_config.json"))["datasets"]
    return (config.keys())


def create_graph(plots: dict, data):

    return dcc.Graph(figure=data)


def generate_plots(plots: dict, plot_type):
    datasets = get_datasets()
    # making two loops for readability
    if plot_type == 'loss':

        frame = dbc.Row(id="loss wrapper", children=[dbc.Row(
            [
                dbc.Col(create_graph(plots, plots[df]["train loss"])),
                dbc.Col(create_graph(plots, plots[df]["val loss"]))
            ]) for df in plots.keys()])
    elif plot_type == 'acc':
        frame = dbc.Row(id="acc wrapper", children=[dbc.Row(
            [
                dbc.Col(create_graph(plots, plots[df]["train acc"])),
                dbc.Col(create_graph(plots, plots[df]["val acc"]))
            ]) for df in plots.keys() if "train acc" in plots[df].keys()])  # messy fix for now to see the acc plots for classification
    else:
        raise Exception('Invalid plot type')
    return frame


def update_plots(plots):
    frame = [
        html.H2('Loss plots'),
        generate_plots(plots, 'loss'),
        html.H2('Accuracy plots'),
        generate_plots(plots, 'acc')]
    return frame


def update_tables(adv=True):

    if adv:

        results = os.listdir(
            f"output{SEPARATOR}results{SEPARATOR}final_results")
        
        results = [r for r in results if "adv" in r]
        name = f"output{SEPARATOR}results{SEPARATOR}final_results{SEPARATOR}"

        columns = ["Norm", "Method", "mean", "std"]
        dtypes = {"Norm": str, "Method": str, "mean": float, "std": float}
        dfs = [pd.read_csv(
            f"{name}{r}", names=columns, dtype=dtypes, header=0) for r in results if "adv" in r]

        # make all floats two decimals
        dfs = [df.round(2) for df in dfs]
        
        frame = [
            dbc.Col(children=[
                # poping the title for tables
                html.H3(f"{results.pop(0).partition('_')[0]}"), #getting the header
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    style_cell={
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 0,
                    },
                    tooltip_data=[
                        {
                            column: {'value': str(value), 'type': 'markdown'}
                            for column, value in row.items()
                        } for row in df.to_dict('records')
                    ],
                    tooltip_duration=None,
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(220, 220, 220)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(210, 210, 210)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    }




                ),
            ]
            )
            for df in dfs
        ]

        return frame


def makehash():
    return collections.defaultdict(makehash)


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            print(key)
            recursive_items(value)
        else:
            print(key)


def plotting(loss_res):
    plots = {}
    for ds in loss_res:
        plots[ds] = {}

        pred_type = loss_res[ds]['pred_type']

        for model in loss_res[ds]:
            if model == 'pred_type':
                continue

            df_train_loss = pd.DataFrame()
            df_val_loss = pd.DataFrame()

            df_train_acc = pd.DataFrame()
            df_val_acc = pd.DataFrame()

            for norm_type in loss_res[ds][model]:
                train_loss = np.mean(
                    loss_res[ds][model][norm_type]['train_loss'], axis=0)
                val_loss = np.mean(loss_res[ds][model]
                                   [norm_type]['val_loss'], axis=0)

                df_train_loss[model + norm_type] = train_loss
                df_val_loss[model + norm_type] = val_loss

                if pred_type == 'classification':
                    train_acc = np.mean(
                        loss_res[ds][model][norm_type]['train_acc'], axis=0)
                    val_acc = np.mean(loss_res[ds][model]
                                      [norm_type]['val_acc'], axis=0)

                    df_train_acc[model + norm_type] = train_acc
                    df_val_acc[model + norm_type] = val_acc

            fig = px.line(df_train_loss)
            fig.update_layout(title_text=ds + ' train loss')
            fig.update_xaxes(title_text='Epochs')
            plots[f"{ds}"]["train loss"] = fig

            fig = px.line(df_val_loss)
            fig.update_layout(title_text=ds + ' val loss')
            fig.update_xaxes(title_text='Epochs')
            plots[f"{ds}"]["val loss"] = fig

            if pred_type == 'classification':
                fig = px.line(df_train_acc)
                fig.update_layout(title_text=ds + ' train acc')
                fig.update_xaxes(title_text='Epochs')
                plots[f"{ds}"]["train acc"] = fig

                fig = px.line(df_val_acc)

                fig.update_layout(title_text=ds + ' val acc train acc')
                fig.update_xaxes(title_text='Epochs')
                plots[f"{ds}"]["val acc"] = fig

                # addint the plots for dash

    return plots


def evaluate(files, output_dir):
    '''
    res = {'ds': {'metric': {'model': ...}}}
    '''
    res = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
    loss_res = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
    print("CURRENT FILE", files)
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
                    
                    res[ds_name][norm_type]['MAE'][model].append(
                        [mean_absolute_error(y_test, y_pred)])
                    res[ds_name][norm_type]['MSE'][model].append(
                        [mean_squared_error(y_test, y_pred)])
                    res[ds_name][norm_type]['R2'][model].append(
                        [r2_score(y_test, y_pred)])
                else:
                    res[ds_name][norm_type]['Acc'][model].append(
                        [accuracy_score(y_test, y_pred)])
                    res[ds_name][norm_type]['AUC'][model].append(
                        [roc_auc_score(y_test, y_pred)])

                if 'train_loss' in output[ds][model].keys():
                    loss_res[ds_name]['pred_type'] = type
                    loss_res[ds_name][model][norm_type]['train_loss'].append(
                        output[ds][model]['train_loss'])
                    loss_res[ds_name][model][norm_type]['val_loss'].append(
                        output[ds][model]['val_loss'])
                    if type == 'classification':
                        loss_res[ds_name][model][norm_type]['train_acc'].append(
                            output[ds][model]['train_acc'])
                        loss_res[ds_name][model][norm_type]['val_acc'].append(
                            output[ds][model]['val_acc'])


    
    # results for csv
    for ds in res:
        with open(output_dir + SEPARATOR + ds +"_adv"+'.csv', 'w') as f: #write to csv
            
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
    return res, loss_res


def make_layout(file_names: list(), all_names: list()):
    """[summary]

    Args:
        file_names ([type]): [description]
        all_names ([type]): [description]

    Returns:
        [type]: [description]
    """
    layout = dbc.Container([
        # store the list of all names for the callback
        dcc.Store(id="all names", data=all_names),
        dbc.Row([
                html.H1('Results Dashboard'),
        dbc.Row([dbc.Container(id="Advanced model wrapper", children=[
            # dbc.Row(id = "loss title",children=html.H1('Loss plots')),
            html.H2("Advanced Models"),
            dbc.Row(id="instruction wrapper", children=[html.P(
                "Select file with configurations: (Epochs,Layers,Batch Size, BN = Batch Norm)")]),
            dcc.Dropdown(
                id="checklist",
                options=[{"label": x, "value": x}
                         for x in file_names if 'adv' in x],  # only show the adv models
                value=file_names[0],
            )]),
            
            ]),
            dbc.Row(id="Table wrapper", children=[
                dbc.Row(id="Adv Tables")
            ]),
            dbc.Row(id="Adv results")
        ]),
        dbc.Row([dbc.Container(id="Simple model wrapper", children=[html.H2("Basic Models")]),
        ])
    ])
    return layout


@app.callback(
    [Output("Adv results", "children"), Output("Adv Tables", "children")],
    [Input("checklist", "value"), Input("all names", "data")])
def update_line_chart(file_name, all_names, prevent_initial_call=False):
    """[summary]

    Args:
        file_names ([type]): [description]
        all_names ([type]): [description]
        prevent_initial_call (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    file_paths = ["output" + SEPARATOR + "results" + SEPARATOR +
                  "predictions" + SEPARATOR + f for f in all_names if file_name in f]  # getting the aggregated versions of adv models


    res, loss_ress = evaluate(
        file_paths, f"output{SEPARATOR}results{SEPARATOR}final_results")
    plots = plotting(loss_ress)
    updated_plots = update_plots(plots)  
    updated_tables = update_tables()
    return updated_plots, updated_tables


if __name__ == '__main__':
   
    # all_names = os.listdir(f"output{SEPARATOR}results{SEPARATOR}predictions")
    #
    # # getting everything until the timestamp
    # grouped_names = list(set([i.partition(")")[0] for i in all_names]))
    #
    # app.layout = make_layout(grouped_names, all_names)
    # app.run_server(debug=True)
    path = '/Users/matej/repos/Normalization-fixaction/output/results/predictions/'
    files = [path + '26112021 155822_basic.pkl',
             path + '29112021 110437_basic.pkl',
             path + '29112021 111536_basic.pkl',
             path + '29112021 112423_basic.pkl',
             path + '29112021 112706_basic.pkl']
    evaluate(files,
             'output')
