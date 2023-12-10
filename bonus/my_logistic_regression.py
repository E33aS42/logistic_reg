import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
import random as rd
from copy import deepcopy


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        try:
            assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[
                1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
            assert np.any(theta), "theta cannot be an empty numpy.ndarray"
            assert isinstance(
                alpha, float), "2nd argument alpha must be a float"
            assert isinstance(
                max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"

            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta

        except Exception as e:
            print(e)
            return None

    def add_intercept(self, x):
        """Adds a column of 1 s to the non-empty numpy.array x.
        Args:
        x: has to be a numpy.array of dimension m * n.
        Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            # assert np.any(x), "argument cannot be an empty numpy.ndarray"
            m = x.shape[0]
            ox = np.ones(m).reshape((m, 1))
            if x.ndim == 1:
                return np.concatenate((ox, x.reshape((m, 1))), axis=1)
            else:
                return np.concatenate((ox, x), axis=1)

        except Exception as e:
            print(e)
            return None

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        try:

            assert isinstance(
                x, np.ndarray) and x.ndim >= 1, "1st argument must be a numpy.ndarray, a vector of dimension m * n"

            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            assert isinstance(self.theta, np.ndarray) and self.theta.ndim == 2 and self.theta.shape[
                0] == x.shape[1] + 1, "2nd argument must be a numpy.ndarray, a vector of dimension n + 1"
            # assert np.any(x) and np.any(theta), "arguments cannot be empty numpy.ndarray"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            X_ = self.add_intercept(x)

            if X_ is not None:
                prod = np.matmul(X_, self.theta)
                return np.array([1 / (1 + exp(-(xi))) for xi in prod]).reshape(-1, 1)
            else:
                print("e")
                return None

        except Exception as e:
            print(e)
            return None

    def loss_(self, y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(
                y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert isinstance(
                y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert isinstance(
                eps, float) and eps > 0, "3rd argument must be a positive float"
            assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
            assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
            m = y.shape[0]
            v_ones = np.ones(y.shape)
            return -1 / m * float(np.matmul(y.T, np.log(y_hat + eps * v_ones)) + np.matmul((v_ones - y).T, np.log(v_ones - y_hat + eps * v_ones)))

        except Exception as e:
            print(e)
            return None

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
        Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"

            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            m = x.shape[0]
            n = x.shape[1]
            # print("x, y, theta: ", x.shape, y.shape, self.theta.shape)
            assert isinstance(self.theta, np.ndarray) and (self.theta.shape == (n + 1, 1) or self.theta.shape == (
                n + 1, )), "theta must be a numpy.ndarray, a vector of dimension (n + 1) * 1"
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            if self.theta.shape == (n + 1, ):
                self.theta = self.theta.reshape(-1, 1)

            m = y.shape[0]
            h = self.predict_(x)
            y_hat = h.reshape(-1, 1)
            sub = y_hat - y
            X = self.add_intercept(x)
            return np.matmul(X.T, sub) / m

        except Exception as e:
            print(e)
            return None

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"

            m = x.shape[0]
            n = x.shape[1]

            step = 0
            x_step = []
            loss_time = []
            while step < self.max_iter:
                # 1. compute loss J:
                J = self.gradient(x, y)

                # 2. update theta:
                self.theta = self.theta - self.alpha * J
                step += 1

                # 3. compute loss for plotting
                x_step.append(step)
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                loss_time.append(self.loss_(y, y_hat))
            x_step = np.array(x_step)
            loss_time = np.array(loss_time)

            return x_step, loss_time

        except Exception as e:
            print(e)
            return None

    def fit_SGD(self, x, y):
        """
        Description:
        Fits the model to the training dataset using Stochastic Gradient Descent.

        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"

            m = x.shape[0]
            n = x.shape[1]

            step = 0
            x_step = []
            loss_time = []

            while step < self.max_iter:
                # 1.select random observation datapoint
                i = rd.randint(0, m - 1)
                xi = x[i].reshape(1, -1)
                yi = y[i].reshape(1, -1)

                # 2. compute loss J:
                J = self.gradient(xi, yi)

                # 3. update theta:
                self.theta = self.theta - self.alpha * J
                step += 1

                # 4. compute loss for plotting
                x_step.append(step)
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                loss_time.append(self.loss_(y, y_hat))
            x_step = np.array(x_step)
            loss_time = np.array(loss_time)

            return x_step, loss_time

        except Exception as e:
            print(e)
            return None

    def fit_minibatch(self, x, y, batch_size=10):
        """
        Description:
        Fits the model to the training dataset using Stochastic Gradient Descent.

        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert isinstance(
                batch_size, int) and batch_size > 0, "3rd argument must be a positive int"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"

            m = x.shape[0]
            n = x.shape[1]

            step = 0
            x_step = []
            loss_time = []

            while step < self.max_iter:
                # 1.select batch_size random observation datapoints
                rdi = np.random.choice(
                    x.shape[0], size=batch_size, replace=False)
                xi = x[rdi, :]
                yi = y[rdi, :]

                # 2. compute loss J:
                J = self.gradient(xi, yi)

                # 3. update theta:
                self.theta = self.theta - self.alpha * J
                step += 1

                # 4. compute loss for plotting
                x_step.append(step)
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                loss_time.append(self.loss_(y, y_hat))
            x_step = np.array(x_step)
            loss_time = np.array(loss_time)

            return x_step, loss_time

        except Exception as e:
            print(e)
            return None

    def fit_momentum(self, x, y, beta=0.9):
        """
        Description:
        A problem with gradient descent is that it can bounce around the search space on optimization problems that have large amounts of curvature or noisy gradients, and it can get stuck in flat spots in the search space that have no gradient.

        Momentum is an extension to the gradient descent optimization algorithm that allows the search to build inertia in a direction in the search space and overcome the oscillations of noisy gradients and coast across flat spots of the search space.
        It adds history to the parameter update equation based on the gradient encountered in the previous updates.

        Momentum involves adding an additional hyperparameter that controls the amount of history (momentum) to include in the update equation, i.e. the step to a new point in the search space.
        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"
            assert isinstance(
                beta, (int, float)) and beta >= 0 and beta <= 1, "3rd argument must be a positive number between 0 and 1"

            m = x.shape[0]
            n = x.shape[1]

            step = 0
            x_step = []
            loss_time = []
            change_1 = np.array([[0] * (n + 1)]).reshape(-1, 1)
            while step < self.max_iter:
                # 1. compute loss J:
                J = self.gradient(x, y)

                # 2. compute momentum change
                change = self.alpha * J + beta * change_1

                # 3. update theta:
                self.theta = self.theta - change
                step += 1
                change_1 = change

                # 4. compute loss for plotting
                x_step.append(step)
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                loss_time.append(self.loss_(y, y_hat))
            x_step = np.array(x_step)
            loss_time = np.array(loss_time)

            return x_step, loss_time

        except Exception as e:
            print(e)
            return None

    def fit_SGD_momentum(self, x, y, beta=0.9):
        """
        Description:
        Fits the model to the training dataset using Stochastic Gradient Descent with momentum.

        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"
            assert isinstance(
                beta, (int, float)) and beta >= 0 and beta <= 1, "3rd argument must be a positive number between 0 and 1"

            m = x.shape[0]
            n = x.shape[1]

            step = 0
            x_step = []
            loss_time = []
            change_1 = np.array([[0] * (n + 1)]).reshape(-1, 1)

            while step < self.max_iter:
                # 1.select random observation datapoint
                i = rd.randint(0, m - 1)
                xi = x[i].reshape(1, -1)
                yi = y[i].reshape(1, -1)

                # 2. compute loss J:
                J = self.gradient(xi, yi)

                # 3. compute momentum change
                change = self.alpha * J + beta * change_1

                # 4. update theta:
                self.theta = self.theta - change
                step += 1
                change_1 = change

                # 5. compute loss for plotting
                x_step.append(step)
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                loss_time.append(self.loss_(y, y_hat))
            x_step = np.array(x_step)
            loss_time = np.array(loss_time)

            return x_step, loss_time

        except Exception as e:
            print(e)
            return None

    @staticmethod
    def score_(y, y_pred):
        try:
            assert isinstance(
                y, np.ndarray) and (y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert isinstance(
                y_pred, np.ndarray) and (y_pred.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
            return sum([1 if yi == yhi else 0 for yi, yhi in zip(y, y_pred)]) / y.shape[0]

        except Exception as e:
            print(e)
