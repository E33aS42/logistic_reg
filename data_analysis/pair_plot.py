import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    try:

        # 1. Load data
        path = "../datasets/dataset_train.csv"
        data_train = pd.read_csv(path)

        labels = list(data_train.select_dtypes(
            include=['int64', 'float64']).columns)

        x_train = data_train[labels]

        # 2. create a distionary of features numpy arrays
        X = {}
        for feature in labels:
            X[feature] = x_train[[feature]].values

        # 3. create pairplot using Seaborn
        labels.remove('Index')
        sns.pairplot(data=data_train[labels + ['Hogwarts House']],
                     hue='Hogwarts House', diag_kind='kde', plot_kws={'s': 2}, height=1.5)

        plt.show()

    except Exception as e:
        print(e)
