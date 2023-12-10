import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from scaler import Standard_Scaler
from logreg_train import num_houses, label_houses, relabel, scatter_plot, mean_
import pickle


# print out whole arrays
np.set_printoptions(threshold=np.inf)

# disable false positive warings
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":
    try:
        # assert len(sys.argv) >= 3, "missing path"

        # 1. Load data
		# path = sys.argv[1]
        path = "../datasets/dataset_train.csv"
        data_train = pd.read_csv(path)
		# path = sys.argv[2]
        path = "../datasets/dataset_test.csv"
        data_testX = pd.read_csv(path)

        labels = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']

        # 2. Replace NaN value by mean
        mean_train = []
        for col in data_train[labels]:
            m = mean_(data_train[col])
            mean_train.append(m)
            data_train[col].fillna(m, inplace=True)
            data_testX[col].fillna(m, inplace=True)
        x = data_train[labels]
        x_train = x.values

        x = data_testX[labels]
        x_test = x.values

        features = labels
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        # 3. Normalization
        # Zscore
        my_Scaler = Standard_Scaler()
        my_Scaler.fit(x_train)
        X_te = my_Scaler.transform(x_test)

        # 4.load models
        # We are going to train 4 logistic regression classifiers to discriminate each class from the others
        with open("models.pickle", "rb") as f:
            models = pickle.load(f)

        # 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.
        y_pred_ = np.array([])
        for i in range(1, 5):
            if y_pred_.any():
                y_pred_ = np.hstack((y_pred_, models[i].predict_(X_te)))
            else:
                y_pred_ = models[i].predict_(X_te)

        # 6. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.
        y_pred = np.argmax(y_pred_, axis=1).reshape(-1, 1) + 1
        # print("fraction of correct predictions for test data:  ", MyLR.score_(y_pred, y_test))	** uncomment if true y values present

        # 8. denumerize predictions
        houses_pred = label_houses(y_pred)
        df = pd.DataFrame(data=houses_pred, columns=['Hogwarts House'])
        df.index.name = 'Index'
        df.to_csv('houses.csv', index=True)

    except Exception as e:
        print(e)
