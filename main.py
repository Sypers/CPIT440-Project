import pandas
from sklearn import linear_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataframe = pandas.read_csv('Real estate.csv')
    x = dataframe.iloc[:, 1:6]
    y = dataframe.iloc[:, 7]
    regression = linear_model.LinearRegression()
    regression.fit(x, y)
    prediction = regression.predict([[2020, 10, 30, 2, 24.8, 121.3]])
    print(prediction)
