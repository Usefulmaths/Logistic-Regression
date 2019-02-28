import numpy as np

def sigmoid(z):
    '''
    Calculates the sigmoid function of 
    an input z.
    '''
    return 1. / (1 + np.exp(-z))


class LogisticRegression(object):
    '''
    A class representing a logistic regression classification model.
    '''

    def __init__(self, trans='quadratic'):
        self.trans = trans
        self.weights = None
        self.losses = None

    def transform(self, X):
        '''
        Transform a feature vector into a higher
        dimensional feature vector

        Arguments:
            X: the feature vector to transforms

        Returns:
            X: the transformed feature vector
        '''
        if self.trans == 'quadratic':
            x_sq = X**2
            x_sq_cross = (X[:, 0] * X[:, 1]).reshape(-1, 1)
            X = np.hstack([X, x_sq])
            X = np.hstack([X, x_sq_cross])
            

        b = np.ones((X.shape[0], 1))
        X = np.hstack([b, X])

        return X

    def fit(self, X, y, lr=0.1, iters=1000, bias=True):
        '''
        Given features and labels of data points, trains
        a logistic regression classifier to the data.

        Arguments:
            X: the features
            y: the labels
            lr: the learning rate
            iters: the number of iterations to perform gradient descent.
            bias: use a bias or not
        '''
        self.losses = []

        if bias:
            X = self.transform(X)

        number_of_features = X.shape[1]
        self.weights = np.random.randn(number_of_features)

        for i in range(iters):
            g = sigmoid(np.dot(X, self.weights))
            gradient = self.loss_derivative(X, y, g)
            self.weights -= lr * gradient

            self.losses.append(self.loss(y, g))

    def loss(self, y, g):
        '''
        Calculates the loss between the true labels
        and the predictions

        Arguments:
            y: the true labels
            g: the predicted labels

        Returns:
            l: the loss between the real and predicted labels.
        '''
        m = y.shape[0]
        e = 10E-50

        l = 1. / m * (-np.dot(y.T, np.log(g + e)) -
                      np.dot((1 - y).T, np.log(1 - g + e)))

        return l

    def loss_derivative(self, X, y, g):
        '''
        Calculates the derivative of the loss
        with respect to the weights.

        Arguments:
            X: the features
            y: the true labels
            g: the predicted labels.
        '''
        m = y.shape[0]
        return 1. / m * np.dot(X.T, (g - y))

    def predict(self, X, bias=True):
        '''
        Predicts the label of given data points.

        Arguments:
            X: the features
            bias: use a bias or not

        Returns: 
            predictions: the predicted labels
        '''
        if bias:
            X = self.transform(X)

        predictions = sigmoid(np.dot(X, self.weights))
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        return predictions

    def accuracy(self, y, y_hat):
        '''
        Calculates the accuracy of the classifier.

        Arguments:
            y: the true labels
            y_hat: the predicted labels

        Returns:
            acc: the accuracy of the classifier
        '''
        number_of_points = y.shape[0]
        equals = y == y_hat

        acc = sum(equals) / number_of_points
        return acc

    def decision_boundary(self, x_range, y_range, number_of_points, bias=True):
        '''
        Generates an array of points that indicate the 
        decision boundary of the classifier

        Arguments:
            x_range: the minimum and maximum x range e.g. [-1, 1]
            y_range: the minimum and maximum y range e.g. [-1, 1]
            number_of_points: the number of points in each range
            bias: use a bias or not
        '''
        X = []

        number_of_points_x = abs(x_range[0] - x_range[1])/number_of_points
        number_of_points_y = abs(y_range[0] - y_range[1])/number_of_points

        for i in np.arange(x_range[0], x_range[1], number_of_points_x):
            for j in np.arange(y_range[0], y_range[1], number_of_points_y):
                X.append([i, j])

        X = np.array(X)

        predictions = self.predict(X, bias=bias)

        return X, predictions