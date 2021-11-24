from helpers import load_data
from gather_results import get_datasets
import numpy as np

dataset = get_datasets()

def get_ds_mean_var(ds):
    df = load_data(ds)
    df = df.select_dtypes(include=[np.float])

    # getting variance of features' mean
    means = list(df.mean())
    outer_var = np.var(means)

    # getting average variance
    var = list(df.var())
    inner_var = np.mean(var)

    return outer_var, inner_var

for ds in dataset:
    outer_var, inner_var = get_ds_mean_var(ds)
    print("--------------------------------------")
    print(f"{ds}")
    print(f"Variance of features' mean: {outer_var}")
    print(f"Average variance: {inner_var}")
