import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from scaler import Standard_Scaler
import pickle
import time

# print out whole arrays
np.set_printoptions(threshold=np.inf)

# disable false positive warnings
pd.options.mode.chained_assignment = None  # default='warn'


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


def label_houses(y):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        dict_ = {}
        dict_[1] = 'Gryffindor'
        dict_[2] = 'Hufflepuff'
        dict_[3] = 'Ravenclaw'
        dict_[4] = 'Slytherin'

        if y.ndim == 2:
            y = y.reshape(-1,)
        y_ = []
        for i in range(len(y)):
            assert y[i] in range(
                1, 5), "house numerized label must be either 1, 2, 3 or 4"
            y_.append(dict_[y[i]])
        return np.array(y_).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None


def relabel(y, fav_label):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(fav_label, int) and fav_label in {
            1, 2, 3, 4}, "2nd argument must be a int that is either 0, 1 ,2 or 3"
        return(np.array([1 if yi[0] == fav_label else 0 for yi in y])).reshape(-1, 1)

    except Exception as e:
        print(e)


def scatter_plot(fig, x1, x2, y_test, y_pred, xlabel, ylabel):
    try:
        assert isinstance(
            x1, np.ndarray) and (x1.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            x2, np.ndarray) and (x2.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_test, np.ndarray) and (y_test.ndim == 1), "3rd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_pred, np.ndarray) and (y_pred.ndim == 1), "4th argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(xlabel, str) and isinstance(
            ylabel, str), "5th, 6th and 7th arguments must be strings"

        fig.scatter(x1[(y_test == 1)], x2[(y_test == 1)], s=200,
                    color='tab:pink', label="true values: Gryffindor")
        fig.scatter(x1[(y_test == 2)], x2[(y_test == 2)], s=200,
                    color='tab:gray', label="true values: Hufflepuff")
        fig.scatter(x1[(y_test == 3)], x2[(y_test == 3)], s=200,
                    color='y', label="true values: Ravenclaw")
        fig.scatter(x1[(y_test == 4)], x2[(y_test == 4)], s=200,
                    color='c', label="true values: Slytherin")
        fig.scatter(x1[(y_pred == 4)], x2[(y_pred == 4)],
                    marker='x', color='b', label="predictions: Slytherin")
        fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], marker='x',
                    color='tab:purple', label="predictions: Gryffindor")
        fig.scatter(x1[(y_pred == 2)], x2[(y_pred == 2)],
                    marker='x', color='g', label="predictions: Hufflepuff")
        fig.scatter(x1[(y_pred == 3)], x2[(y_pred == 3)], marker='x',
                    color='tab:brown', label="predictions: Ravenclaw")
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel)
        fig.grid()
        # fig.legend()

    except Exception as e:
        print(e)


def mean_(x):
    try:
        assert isinstance(
            x, pd.Series), "argument must be a panda series or dataframe"
        column_list = x.tolist()
        m = 0
        cnt = 0
        for i in range(len(column_list)):
            if str(column_list[i]) != 'nan':
                m += column_list[i]
                cnt += 1
        m /= cnt
        return m

    except Exception as e:
        print(e)


def mean_xy(x, y):
    try:
        assert isinstance(
            x, pd.Series), "argument must be a panda series or dataframe"
        x_ = x.tolist()

        m = {}
        cnt = {}
        for h in range(1, 5):
            m[h] = 0
            cnt[h] = 0
        for i in range(len(x_)):
            if str(x_[i]) != 'nan':
                m[y[i][0]] += x_[i]
                cnt[y[i][0]] += 1
        for h in range(1, 5):
            m[h] /= cnt[h]
        return m

    except Exception as e:
        print(e)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            GD = "batch"
        else:
            GD = sys.argv[1]
        assert GD in {"batch", "sgd", "minibatch", "momentum", "sgdmom"}, "wrong argument, allowed keywords are:\n\
		\"batch\": basic gradient descent algorithm\n\
		\"sgd\": stochastic gradient descent algorithm\n\
		\"minibatch\": minibatch gradient descent algorithm\n\
		\"momentum\":  basic gradient descent algorithm with momentum\n\
		\"sgdmom\": stochastic gradient descent algorithm with momentum\n\
		"

        # 1. Load data
        path = "../datasets/dataset_train.csv"
        data_train = pd.read_csv(path)
        # labels = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
        labels = list(data_train.select_dtypes(
            include=['int64', 'float64']).columns)
        labels.remove('Index')
        labels.remove('Arithmancy')
        labels.remove('Potions')
        labels.remove('Care of Magical Creatures')
        labels.remove('History of Magic')
        labels.remove('Transfiguration')
        labels.remove('Divination')
        labels.remove('Muggle Studies')
        labels.remove('Flying')
        labels.remove('Defense Against the Dark Arts')
        labels.remove('Herbology')

        print(labels)

        # 2. numerize y labels
        y = data_train[['Hogwarts House']].values
        y_train = num_houses(y)

        # Replace NaN value by mean
        x = data_train[labels]
        m = {}
        for col in x:
            m[col] = mean_xy(x[col], y_train)
        for col in x:
            for i in range(len(x)):
                if str(x[col].iloc[i]) == 'nan':
                    x[col].iloc[i] = m[col][y_train[i][0]]
        x_train = x.values

        features = labels
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        colors = ['r', 'y', 'b', 'g']

        # 3. Normalization
        # Zscore
        my_Scaler = Standard_Scaler()
        my_Scaler.fit(x_train)
        X_tr = my_Scaler.transform(x_train)

        # 4. Training
        # We are going to train 4 logistic regression classifiers to discriminate each class from the others
        time_init = time.time()
        models = {}
        y_ = {}
        fig = plt.figure()

        for i in range(1, 5):
            # 4.a relabel y labels
            y_[i] = relabel(y_train, i)

            # 4.b training
            models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)),
                             alpha=1e-1, max_iter=400)
            if GD == "batch":
                x_step, loss_time = models[i].fit_(X_tr, y_[i])
            elif GD == "sgd":
                x_step, loss_time = models[i].fit_SGD(X_tr, y_[i])
            elif GD == "minibatch":
                bs = 10
                if len(sys.argv) == 3:
                    bs = sys.argv[2]
                    assert bs.isdigit(), "2nd argument must be a positive integer"
                    bs = int(bs)
                x_step, loss_time = models[i].fit_minibatch(
                    X_tr, y_[i], batch_size=bs)
            elif GD == "momentum":
                beta = 0.9
                if len(sys.argv) == 3:
                    beta = sys.argv[2]
                    beta = float(beta)
                    assert beta >= 0 and beta <= 1,  "3rd argument must be a positive number between 0 and 1"
                models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)),
                                 alpha=1e-1, max_iter=40)
                x_step, loss_time = models[i].fit_momentum(
                    X_tr, y_[i], beta=beta)
            elif GD == "sgdmom":
                beta = 0.9
                if len(sys.argv) == 3:
                    beta = sys.argv[2]
                    beta = float(beta)
                    assert beta >= 0 and beta <= 1,  "3rd argument must be a positive number between 0 and 1"
                models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)),
                                 alpha=1e-1, max_iter=40)
                x_step, loss_time = models[i].fit_SGD_momentum(
                    X_tr, y_[i], beta=beta)
            # print(models[i].theta)
            plt.plot(x_step, loss_time,
                     label=houses[i-1], linewidth=2, c=colors[i-1])
        time_end = time.time()
        print("execution time: ", round(time_end - time_init, 2))
        fig.suptitle(GD + "\nLoss over time\n" + "Execution time: " +
                     str(round(time_end - time_init, 2)) + "s")
        plt.grid()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("loss_with_" + GD + ".png")

        # 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.

        y_pred_tr_ = np.array([])
        for i in range(1, 5):
            if y_pred_tr_.any():
                y_pred_tr_ = np.hstack((y_pred_tr_, models[i].predict_(X_tr)))
            else:
                y_pred_tr_ = models[i].predict_(X_tr)

        # # 6. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.

        y_pred_tr = np.argmax(y_pred_tr_, axis=1).reshape(-1, 1) + 1
        print("fraction of correct predictions for train data:  ",
              MyLR.score_(y_pred_tr, y_train))

        # # 7. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
        ax, fig = plt.subplots(1, sum(range(len(labels))),
                               figsize=(30, 10), constrained_layout=True)
        cnt = set()
        k = 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and ((i, j) not in cnt) and i < j:
                    cnt.add((i, j))
                    scatter_plot(fig[k], x_train[:, i], x_train[:, j], y_train.reshape(
                        -1,), y_pred_tr.reshape(-1,), labels[i], labels[j])
                    k += 1
        fig[k - 1].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n"
                     + "Percentage of correct predictions:  " + str(round(100 * MyLR.score_(y_pred_tr, y_train), 1)) + "%\n" + "labels: " + str(labels))
        plt.show()

        # 8. Save models
        with open("models.pickle", "wb") as f:
            pickle.dump(models, f)

    except ValueError:
        print("input is not a number")
    except Exception as e:
        print(e)
