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


def check_input(data, features):
    try:
        assert isinstance(data, pd.DataFrame), "Dataframe Error"
        assert isinstance(
            features, list) and features, "second argument should be a list of non-empty strings"
        for v in features:
            assert isinstance(
                v, str) and v, "second argument should be a list of non-empty strings"
        return True
    except AssertionError as msg:
        print(msg)
        return False
    except Exception as msg:
        print(msg)
        return False


if __name__ == "__main__":
    try:

        # 1. Load data
        path = "../datasets/dataset_train.csv"
        data_train = pd.read_csv(path)

        labels = data_train.select_dtypes(include=['int64', 'float64']).columns
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        # print(labels)
        x_train = data_train[labels]
        y_train = data_train[['Hogwarts House']].values

        # 2. numerize y labels
        Y_tr = num_houses(y_train)

        X = {}
        for feature in labels:
            X[feature] = x_train[[feature]].values

        _, fig = plt.subplots(2, int(len(labels)/2), figsize=(25, 20))
        i = 0
        j = 0
        for feature in labels:
            sns.histplot(X[feature][(Y_tr == 1)], bins=20,
                         color='r', ax=fig[i][j], kde=True)
            sns.histplot(X[feature][(Y_tr == 2)], bins=20,
                         color='y', ax=fig[i][j], kde=True)
            sns.histplot(X[feature][(Y_tr == 3)], bins=20,
                         color='b', ax=fig[i][j], kde=True)
            sns.histplot(X[feature][(Y_tr == 4)], bins=20,
                         color='g', ax=fig[i][j], kde=True)

            fig[i][j].set_xlabel(feature)
            fig[i][j].set_ylabel("")
            fig[i][j].legend(houses)
            # fig[i][j].grid()
            j += 1
            if j == int(len(labels)/2):
                i = 1
                j = 0
        plt.show()

        # Question: Which Hogwarts course has an homogeneous score distribution between all four houses?
        # Answer: Arithmancy and Care of magical creatures appear to have an homogeneous score distribution between all four houses

    except Exception as e:
        print(e)
