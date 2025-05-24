import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import corrected correlation matrix


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def correlation_matrix(data):

    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig('result/correlation_matrix.png')

    return corr

def plot_histogram(data):
    """
    Plot a histogram of each column in the data.
    Group all plots in one figure.
    """
    columns = data.columns.tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, column in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
    plt.tight_layout()
    plt.savefig('result/histogram.png')
    plt.close()
    
def detect_outliers(data):
    """
    Detect outliers in a specified column using the IQR method.
    """
    columns = data.columns.tolist()
    outliers = {}
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers

def plot_boxplot(data ):
    """
    Plot a boxplot of a specified column in the data.
    group all plots in one figure
    """
    columns = data.columns.tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, column in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
    plt.tight_layout()
    plt.savefig('result/boxplot.png')
    plt.close()


def drop_correlated_features(data, threshold=0.8):
    """
    Drop features that are highly correlated with each other.
    """
    data = data.drop("label", axis=1)
    corr = data.corr()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                colname = corr.columns[i]
                to_drop.add(colname)
    data.drop(columns=to_drop, inplace=True)
    return data





