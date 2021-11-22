from helpers import load_data
from gather_results import get_datasets
import numpy as np

dataset = get_datasets()

def get_ds_mean_var(ds):
    df = load_data(ds)
    df = df.select_dtypes(include=[np.float])
    means = list(df.mean())
    var = np.var(means)
    return var

for ds in dataset:
    var = get_ds_mean_var(ds)
    print(f"{ds}: {var}")
