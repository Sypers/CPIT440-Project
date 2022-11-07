# Make sure that you have the following libraries installed
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model

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
    # Split the data into train and test sets for x and y
    x_train, x_test, y_train, y_test = train_test_split(dataframe.iloc[:, 1:7], dataframe.iloc[:, 7], test_size=0.024, shuffle=False)
    # apply linear regression to dataset
    regression = linear_model.LinearRegression()
    regression.fit(x_train, y_train)  # learning process
    # Create a series to save prediction values
    prediction = pandas.Series(regression.predict(x_test), index=x_test.index, name="Y Prediction")
    results = pandas.concat([y_test, prediction], axis=1)
    print(results)
    print("The Accuracy of the linear regression model = ",  round(regression.score(x_test, y_test)*100, 4), "%")
    # The Scatter points for dataset values represented with blue dots
    plt.scatter(y_test.index, y_test)
    # scatter point for the predicted value represented with a red dot
    plt.scatter(prediction.index, prediction, c='r')
    # plot the linear regression line on the scatter plot
    m, b = np.polyfit(y_test.index, y_test, 1)
    plt.plot(y_test.index, m * y_test.index + b, color='green')
    # show the completed scatter plot
    plt.show()

