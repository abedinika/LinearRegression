"""
Designed and developed by: Nika Abedi - contact email: nikka.abedi@gmail.com
*****************************************************************************
Linear Regression Class
*****************************************************************************

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


class LinRegression:

    model = []

    # -------------------------------------------------------------
    # Class constructor
    # -------------------------------------------------------------
    def __init__(self):
        # create my data set
        self.y = np.arange(1, 1000)
        self.x = (np.arange(1, 1000)) * 0.1

        # Split the data into training/testing sets
        self.x_train = self.x[:-300]
        self.x_test = self.x[-300:]

        self.x_train = np.expand_dims(self.x_train, 1)
        self.x_test = np.expand_dims(self.x_test, 1)

        self.y_train = self.y[:-300]
        self.y_test = self.y[-300:]

        self.y_train = np.expand_dims(self.y_train, 1)
        self.y_test = np.expand_dims(self.y_test, 1)

        self.LR()
    # -------------------------------------------------------------
    # objective function
    # -------------------------------------------------------------
    def objfunction(self):
        for i in range(np.size(self.x)):
            self.y[i] = 2 * self.x[i] + 0.1
        return self.y

    # -------------------------------------------------------------
    # Linear Regression model
    # -------------------------------------------------------------
    def LR(self):
        # Create linear regression model
        self.model = linear_model.LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    # -------------------------------------------------------------
    # Test phase
    # -------------------------------------------------------------
    def Test(self):
        # Make predictions using the testing set
        pred = self.model.predict(self.x_test)
        return pred

    # -------------------------------------------------------------
    # Plot the outputs
    # -------------------------------------------------------------
    def main(self):

        # The coefficients
        print('Coefficients: \n', self.model.coef_)

        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, self.Test()))

        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(self.y_test, self.Test()))

        # Scatter test data
        plt.scatter(self.x_test, self.y_test, color='red')
        # Plot predicted test data
        plt.plot(self.x_test, self.Test(), color='blue', linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()