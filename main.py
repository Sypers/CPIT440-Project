import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib
# Press the green button in the gutter to run the script.
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
    dataframe = pandas.read_csv('Real estate.csv')
    # Gather columns from dataset
    x = dataframe.iloc[:, 1:7]  # variables
    y = dataframe.iloc[:, 7]  # value to fit
    # apply linear regression to dataset
    regression = linear_model.LinearRegression()
    regression.fit(x.values, y)  # learning process
    # use user inputs to predict using custom values
    # x1 = float(input("enter transaction date e.g: (2013.55): "))
    # x2 = float(input("enter house age: "))
    # x3 = float(input("enter distance to the nearest mass rapid transit system in meters: "))
    # x4 = float(input("number of convenience stores nearby: "))
    # x5 = float(input("enter house latitude (between 24.9 - 25): "))
    # x6 = float(input("enter house longitude (between 121-122): "))
    x1 = 2013
    x2 = 10
    x3 = 700
    x4 = 1
    x5 = 24.94
    x6 = 121.6

    # predict the value
    prediction = regression.predict([[x1, x2, x3, x4, x5, x6]])
    # print predicted value
    print("----------------------------------------------------------\nThe predicted house price per unit area is ",
          prediction)
    plt.scatter(dataframe.iloc[:, 0], y)  # The Scatter points for dataset values represented with blue dots
    plt.scatter(len(dataframe)+1, prediction, c='r')  # scatter point for the predicted value represented with a red dot
    plt.show()  # show the completed scatter plot
