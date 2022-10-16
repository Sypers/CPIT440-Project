import pandas
from sklearn import linear_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataframe = pandas.read_csv('Real estate.csv')
    x = dataframe.iloc[:, 1:7]
    y = dataframe.iloc[:, 7]
    regression = linear_model.LinearRegression()
    regression.fit(x.values, y)
    x1 = float(input("enter transaction date e.g: (2013.55): "))
    x2 = float(input("enter house age: "))
    x3 = float(input("enter distance to the nearest mass rapid transit system: "))
    x4 = float(input("number of convenience stores nearby: "))
    x5 = float(input("enter house latitude (between 24.9 - 25): "))
    x6 = float(input("enter house longitude (between 121-122): "))
    prediction = regression.predict([[x1, x2, x3, x4, x5, x6]])
    print("----------------------------------------------------------\nThe predicted house price per unit area is ", prediction)
