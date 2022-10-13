import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from plotly import figure_factory as ff
import statsmodels.api
from plotly import express as px
import numpy as np
import pandas as pd
import seaborn as sns
from plotly import graph_objects as go
from sklearn.inspection import permutation_importance


def check_response_type(titanic_df, response):
    if len(pd.unique(titanic_df[response])) >= 2:
        print("Response variable is boolean")
        boolean_check = True
    else:
        print("Response variable is continuous")
        boolean_check = False
    return boolean_check


def check_predictor_type(titanic_df, predict_name):
    if len(pd.unique(titanic_df[predict_name])) <= 3:
        cat_check = True
        print(predict_name + " variable is categorical")

    else:
        cat_check = False
        print(predict_name + "  variable is continuous")
    return cat_check


def cat_response_cont_predictor(titanic_df, predictor, response):

    responses = titanic_df[response].unique()
    print(responses)
    return


def cont_response_cat_predictor(titanic_df, predictor, response):
    predictors = titanic_df[predictor].unique()
    print(predictors)

    return


def cont_response_cont_predictor(titanic_df, predictor_name, response):
    n = 200
    x = titanic_df[predictor_name]
    y = titanic_df[response]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()

    return


def cat_response_cat_predictor(titanic_df, predictor_name, response):
    n = 200
    x = titanic_df[predictor_name]
    y = titanic_df[response]

    x_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()

    x = np.random.randn(n)
    y = x + np.random.randn(n)

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()

    return


def linear_regression(titanic_df, predictor_name, response):
    x = titanic_df[predictor_name]
    y = titanic_df[response]

    predictor = statsmodels.api.add_constant(x)
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {predictor_name}")
    print(linear_regression_model_fitted.summary())
    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
    # Plot the figure
    fig = px.scatter(x=x, y=y, trendline="ols")

    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",

    )

    fig.show()


def logistic_regression(titanic_df, predictor_name, response):
    x = titanic_df[predictor_name]
    y = titanic_df[response]

    predictor = statsmodels.api.add_constant(x)
    logistic_regression_model = statsmodels.api.OLS(y, predictor)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(f"Variable: {predictor_name}")
    print(logistic_regression_model_fitted.summary())

    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=x, y=y, trendline="ols")

    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",

    )

    fig.show()


def main():
    titanic_df = pd.read_csv('/home/osboxes/BDA_Repository/BDA_602/titanic_train.csv')
    titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
    titanic_df.drop('Cabin', axis=1, inplace=True)
    titanic_df.drop('Name', axis=1, inplace=True)
    print(titanic_df.isna().sum())
    print(titanic_df.head())
    titanic_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    titanic_df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    titanic_df['Embarked'].fillna(titanic_df['Embarked'].mean(), inplace=True)
    print(titanic_df.head())
    predictors = [
        "Pclass",
        "Sex",
        "Age",
        "Embarked",
        "Fare",
        "Parch",
    ]

    response = "Survived"
    # Determine if response is continuous or boolean
    boolean_check = check_response_type(titanic_df, response)

    for predictor in predictors:
        # Determine if the predictor is cat/cont
        cat_check = check_predictor_type(titanic_df, predictor)
        x = titanic_df[predictor]
        y = titanic_df[response]

        if boolean_check is False and cat_check is False:
            cont_response_cont_predictor(titanic_df, predictor, response)
        elif boolean_check is True and cat_check is False:
            cat_response_cont_predictor(titanic_df, predictor, response)
        elif boolean_check is False and cat_check is False:
            cont_response_cat_predictor(titanic_df, predictor, response)
        else:
            cat_response_cat_predictor(titanic_df, predictor, response)

        if cat_check is False:
            if boolean_check is True:
                logistic_regression(titanic_df, predictor, response)
            else:
                linear_regression(titanic_df, predictor, response)


if __name__ == "__main__":
    main()
