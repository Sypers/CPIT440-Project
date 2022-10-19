import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib

# Press the green button in the gutter to run the script.
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    This is an example of using linear regression on a dataset to predict a house price per unit area
    The dataset consists of the following variables:
    x1 = transaction date
    x2 = house age in number of years
    x3 = distance to the nearest MRT(Mass rapid transport) station
    x4 = number of convenience stores nearby
    x5 = house latitude location value
    x6 = house longitude location value
    y = house price value per unit area
    """
    # Gather columns from dataset
    dataframe = pandas.read_csv('Real estate.csv')
    x_train, x_test, y_train, y_test = train_test_split(dataframe.iloc[:, 1:7], dataframe.iloc[:, 7], test_size=0.025, shuffle=False)
    # apply linear regression to dataset
    regression = linear_model.LinearRegression()
    regression.fit(x_train, y_train)  # learning process
    # Create a series to save prediction values
    prediction = pandas.Series(regression.predict(x_test), index=x_test.index, name="Y Prediction")
    results = pandas.concat([y_test, prediction], axis=1)
    print(results)
    print("The Accuracy of the linear regression model = ",  round(regression.score(x_test, y_test)*100, 4), "%")
    # predict the value
    # predictions = regression.predict()
    # print predicted value
    # print("----------------------------------------------------------\nThe predicted house price per unit area is ",
    #       prediction)
    # The Scatter points for dataset values represented with blue dots
    # plt.scatter(dataframe.iloc[:, 0], y)
    # scatter point for the predicted value represented with a red dot
    # plt.scatter(len(dataframe), prediction, c='r')
    # plot the linear regression line on the scatter plot
    # m, b = np.polyfit(dataframe.iloc[:, 0], y, 1)
    # plt.scatter(dataframe.iloc[:, 0], m * dataframe.iloc[:, 0] + b)
    # show the completed scatter plot
    # plt.show()

