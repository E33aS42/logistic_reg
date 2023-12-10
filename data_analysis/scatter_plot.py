import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
import numpy.random as rd


def num_houses(y):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        dict_ = {}
        dict_['Gryffindor'] = 1
        dict_['Hufflepuff'] = 2
        dict_['Ravenclaw'] = 3
        dict_['Slytherin'] = 4

        houses = {'Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'}

        y_ = []
        for i in range(len(y)):
            assert y[i][0] in houses, "house name not recognized"
            y_.append(dict_[y[i][0]])
        return np.array(y_).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    try:

        # 1. Load data
        path = "../datasets/dataset_train.csv"
        data_train = pd.read_csv(path)

        labels = data_train.select_dtypes(include=['int64', 'float64']).columns
        # print(labels)
        x_train = data_train[labels]
        y_train = data_train[['Hogwarts House']].values

        # 2. numerize y labels
        Y_tr = num_houses(y_train)

        # 3. create a distionary of features numpy arrays
        X = {}
        for feature in labels:
            X[feature] = x_train[[feature]].values

        # 4. Scatter plots (check if it can be done better with seaborn)
        for f1 in labels:

            i = 0
            j = 0
            ax, fig = plt.subplots(2, int(len(labels)/2), figsize=(25, 20))
            ax.tight_layout(pad=5.0)
            for f2 in labels:
                fig[i][j].scatter(X[f1][(Y_tr == 1)], X[f2][(
                    Y_tr == 1)], marker='o', c='r', label='Gryffindor')
                fig[i][j].scatter(X[f1][(Y_tr == 2)], X[f2][(
                    Y_tr == 2)], marker='o', c='y', label='Hufflepuff')
                fig[i][j].scatter(X[f1][(Y_tr == 3)], X[f2][(
                    Y_tr == 3)], marker='o', c='b', label='Ravenclaw')
                fig[i][j].scatter(X[f1][(Y_tr == 4)], X[f2][(
                    Y_tr == 4)], marker='o', c='g', label='Slytherin')

                fig[i][j].set_xlabel(f1)
                fig[i][j].set_ylabel(f2)
                fig[i][j].legend()
                fig[i][j].grid()
                j += 1
                if j == int(len(labels)/2):
                    i = 1
                    j = 0

        plt.show()

        # Question: What are the two features that are similar ?
        # Answer: Looking at the different graphics, we can see that the relation between Astronomy vs. Defense Against the Dark Art is linear which means that they are similar.

    except Exception as e:
        print(e)
