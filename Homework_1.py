# importing libraries

import pandas as pd
import plotly.express as px
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. loading the iris data using Pandas DataFrame
iris = pd.DataFrame()
x_scaled = []
x = []
y = []


def load_database():
    global iris
    iris_data = datasets.load_iris()
    print(iris_data)
    print(iris_data.target_names)
    print(iris_data.feature_names)
    print(iris_data.data[0:5])
    print(iris_data.target)
    iris = pd.DataFrame(
        {
            "sepal length": iris_data.data[:, 0],
            "sepal width": iris_data.data[:, 1],
            "petal length": iris_data.data[:, 2],
            "petal width": iris_data.data[:, 3],
            "species": iris_data.target,
        }
    )
    print(iris)


# 2. getting sum simple summary statistics (mean, min, max) using numpy
def numpy_statistics():
    global iris
    n_array = iris.to_numpy()
    print(n_array)
    mean_value = n_array.mean(axis=0)
    min_value = n_array.min(axis=0)
    max_value = n_array.max(axis=0)

    print("mean = ", mean_value)
    print("min = ", min_value)
    print("max = ", max_value)


# 3. Different types of plots (Histogram, scatter plot, violin, boxplot, barchart)


def plot_graphs():
    global iris
    histogram = px.histogram(iris, x="petal length")
    histogram.show()

    scatter_plot = px.scatter(iris, x="sepal length", y="sepal width")
    scatter_plot.show()

    violin_plot = px.violin(iris, x="sepal length", color="species")
    violin_plot.show()

    box_plot = px.box(iris, x="petal length", y="species")
    box_plot.show()

    bar_chart = px.bar(
        iris,
        x="petal length",
        y="petal width",
        color="species",
    )
    bar_chart.show()


# 4. standardscaler transformer using sk learn
def standardization():
    global x_scaled
    global iris
    s = iris.iloc[:, [0, 1, 2, 3]]
    s.head()
    scaler = StandardScaler().fit(s)
    print(scaler)
    x_scaled = scaler.transform(s)
    print(x_scaled)


# 5. fitting transformed data against random forest classifier
def random_forest():
    x = x_scaled[:, [0, 1, 2, 3]]
    y = iris["species"]
    # Splitting the data into training and testing data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=100)
    # train the model using training sets
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # finding the accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def main():
    load_database()
    numpy_statistics()
    plot_graphs()
    standardization()
    random_forest()


if __name__ == "__main__":
    main()
