import logging
import numpy as np
import matplotlib.pyplot as plt
from sys import platform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from helpers import load_data

try:
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    pass

SEPARATOR = '\\' if platform == 'win32' else '/'


def without_label(data, config):
    """Removing label and categorical values"""
    data_without_cat = data.select_dtypes(exclude='category')
    target = config["target"]
    X = data_without_cat.loc[:, data_without_cat.columns != target]  # removing label
    return X


def z_score(data, config):
    data_copy = data.copy()
    data_to_normalize = without_label(data_copy, config)
    scaler = StandardScaler()
    data_copy[data_to_normalize.columns] = scaler.fit_transform(data_to_normalize)

    return data_copy


def tanh_norm(data, config):
    # Reduce influence of the values in the tail of the distribution
    # x = 0.5 * (tanh((0.01(x - mu)) / std) + 1)

    data_copy = data.copy()
    data_to_normalize = without_label(data_copy, config)

    m = np.mean(data_to_normalize, axis=0)
    std = np.std(data_to_normalize, axis=0)

    data_copy[data_to_normalize.columns] = 0.5 * (np.tanh(0.01 * ((data_to_normalize - m) / std)) + 1)
    return data_copy


def min_max(data, config):
    data_copy = data.copy()
    data_to_normalize = without_label(data_copy, config)

    scaler = MinMaxScaler()
    data_copy[data_to_normalize.columns] = scaler.fit_transform(data_to_normalize)
    return data_copy


def pareto_scaling(data, config):
    data_copy = data.copy()
    data_to_normalize = without_label(data_copy, config)

    mean = np.mean(data_to_normalize, axis=0)
    std = np.std(data_to_normalize, axis=0)

    data_copy[data_to_normalize.columns] = (data_to_normalize - mean) / np.sqrt(std)
    return data_copy


def variable_scaling(data, config):
    # extends the z-score normalization by introducing the Co- efficient of Variation (CV) as a scaling factor.
    # The coefficient of variation is given as the ratio of the mean of data to its standard deviation
    data_copy = data.copy()
    data_to_normalize = without_label(data_copy, config)

    mean = np.mean(data_to_normalize, axis=0)
    std = np.std(data_to_normalize, axis=0)

    data_copy[data_to_normalize.columns] = ((data_to_normalize - mean) * mean) / (std ** 2)
    return data_copy


class Normalizator:
    def __init__(self, dataset: str):
        """
        Class that reads data and performs normalization methods
        Args:
            - dataset : str of the type of dataset to use, not that this also needs to be implemented as a separete field in load_data
        Methods:
            - normalize: function to normalize the data given method
            - boxplot: creates subplots of all variables
            
        """

        self.dataset = dataset
        self.df, self.config = load_data(self.dataset, get_config=True)
        self.df_norm = None

    def normalize(self, method="zscore", reset=False, save=False):
        # just to make sure the data is not tampered with before normalization
        # if reset is true, the data is reset to the original state
        # if save is true, the data is saved to a new file in the post_norma_data folder, the relative path
        logging.info(f'Normalizing with {method}')
        if reset:
            # Load the data again
            self.df = load_data(self.dataset)
        if method == "zscore":
            self.df_norm = z_score(self.df, self.config)
        elif method == "tanh":
            self.df_norm = tanh_norm(self.df, self.config)
        elif method == "pareto":
            self.df_norm = pareto_scaling(self.df, self.config)
        elif method == 'minmax':
            self.df_norm = min_max(self.df, self.config)
        elif method == 'variablescaling':
            self.df_norm = variable_scaling(self.df, self.config)
        else:
            print(method)
            raise NotImplementedError("This normalization is not implemented yet")

        if save:
            # save the data to a new file
            self.df_norm.to_csv(rf"output{SEPARATOR}post_norma_data{SEPARATOR}{self.dataset}_{method}.csv", index=False)

    def visuals(self):
        fig, axes = plt.subplots(3, 4, figsize=(15, 15))
        for i, el in enumerate(list(self.df.columns.values)[:-1]):
            a = self.df.boxplot(el, ax=axes.flatten()[i])

        b = self.df.hist(figsize=(15, 15))
        plt.tight_layout()

        plt.show()

    def label_encoder(self, col):
        le = LabelEncoder()
        le.fit(self.df[col])
        self.df[col] = le.transform(self.df[col])
        return self.df

# if __name__ == "__main__":
#     n = Normalizator("adult")
#     n.normalize(method="minmax", save=True)
#     print(n.df.describe())
#     print(n.df_norm.describe())
#     # n.visuals()

# n.run_model(model="knn")
