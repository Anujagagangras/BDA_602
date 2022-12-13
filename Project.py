import os
from os.path import abspath, dirname, join
import pandas as pd
from itertools import combinations
import sqlalchemy
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, tree
from sklearn.metrics import plot_confusion_matrix
from plotly import express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from plotly.subplots import make_subplots
from plotly import graph_objects as go


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def import_data():
    db_user = "root"
    db_pass = "osboxes.org"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)
    # importing result table from baseball.sql which contains required features
    query = """
            SELECT *
            FROM rolling_final_joint
            """

    # converting sql table into Pandas dataframe.
    df = pd.read_sql_query(query, sql_engine)

    # response variable Home_Team_Wins
    response = "HomeTeamWins"
    # drop rows which contain missing values
    df.dropna(axis=0)
    return df


def inspect_data(df):
    # print data shape
    print(df.shape)
    # Check Top 5 rows
    print(df.head())
    # Check datatype
    print(df.dtypes)
    # Check data info
    print(df.info())
    # to Get data summary
    print(df.describe())
    # Checking for NULL values
    print(df.isnull().sum())
    # Check Correlation
    print(df.corr())
    # drop rows which contain missing values
    df.dropna(axis=0)


def clean_data(_df):
    _df["local_date"] = pd.to_datetime(_df["local_date"])
    _df = _df.dropna()
    print(_df.isnull().values.sum())
    print(_df.shape)
    return _df


def train_test_split(_df):
    train = _df.loc[_df["local_date"] < "2010-12-12"]
    test = _df.loc[_df["local_date"] >= "2010-12-12"]
    X_train = train.loc[:, ~test.columns.isin(['HomeTeamWins', 'local_date'])]
    X_test = test.loc[:, ~test.columns.isin(['HomeTeamWins', 'local_date'])]
    y_train = train['HomeTeamWins']
    y_test = test['HomeTeamWins']

    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, y_train, X_test, y_test):
    x = X_train
    y = y_train

    logistic_regression_model = sm.Logit(y, x)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(logistic_regression_model_fitted.summary())

    # performing predictions on the test dataset
    yhat = logistic_regression_model_fitted.predict(X_test)
    prediction = list(map(round, yhat))

    # comparing original and predicted values of y
    # print('Actual values', list(y_test.values))
    # print('Predictions :', prediction)
    # confusion matrix
    cm = confusion_matrix(y_test, prediction)
    print("Confusion Matrix : \n", cm)

    y_pred = logistic_regression_model_fitted.predict(X_test)

    # Evaluating the Algorithm

    print("\n\rLogistic regression\n\r")
    # accuracy score of the model
    print('Logistic Regression Accuracy', accuracy_score(y_test, prediction))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def random_forest(X_train, y_train, X_test, y_test):
    RF_clf = RandomForestClassifier(max_depth=19, random_state=0)
    RF_Fitted = RF_clf.fit(X_train, y_train)
    y_pred_RF = RF_Fitted.predict(X_test)

    y_pred_RF = RF_Fitted.predict(X_test)
    accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)

    # Evaluating the Algorithm

    print("\n\rFor Random Forest\n\r")
    print("Random Forest Accuracy:", accuracy_RF)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_RF))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_RF))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))

    return RF_Fitted


def decision_tree(X_train, y_train, X_test, y_test):
    DT_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2, min_samples_split=2)
    DT_Fitted = DT_clf.fit(X_train, y_train)
    # print(DT_clf.feature_importances_)
    y_pred_DT = DT_Fitted.predict(X_test)
    accuracy_DT = metrics.accuracy_score(y_test, y_pred_DT)

    print("\n\rDecision Tree\n\r")
    print("Decision Tree Accuracy:", accuracy_DT)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_DT))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_DT))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_DT)))


def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


def naive_bayes_model(X_train, y_train, X_test, y_test):
    # training the model on training set
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_NB = gnb.predict(X_test)
    accuracy_NB = metrics.accuracy_score(y_test, y_pred_NB)

    print("\n\r Naive Bayes \n\r")
    print("Naive Bayes Accuracy:", accuracy_NB)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_NB))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_NB))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_NB)))


def mean_of_response(df, predictors, response):
    df = df.dropna()

    MOR_table = pd.DataFrame(columns=[
        'Response',
        'Predictor',
        'Mean_Squared_Error',
        'Mean_Squared_Error_Weighted',
        'URL',
    ])

    for i in range(0, len(predictors)):

        bin_size = 10
        if len(df[predictors[i]].unique()) < 6:
            bin_size = len(df[predictors[i]].unique())

        resp = df[response]

        df_copy = df[[predictors[i], response]].copy()

        count, bins_edges = np.histogram(df_copy[predictors[i]], bins=bin_size)

        MOR_Table = pd.DataFrame()
        MOR_Table["Bin_Count"] = count

        bins_weighted_mean_squared_diff = [0 for i in range(bin_size)]
        bins_unweighted_mean_squared_diff = [0 for i in range(bin_size)]
        bin_mean = [0 for i in range(bin_size)]
        print()

        pop_mean = np.mean(df_copy[response])
        print("population mean for predictor = ", pop_mean)

        for i_bin in range(bin_size):
            bin_data = df_copy[
                (bins_edges[i_bin + 1] >= df_copy[predictors[i]]) & (bins_edges[i_bin] < df_copy[predictors[i]])]
            # step 5.Difference with mean of response along with its plot (weighted and unweighted)
            # print(bin_data[predictor])
            # print()
            # weighted mean or response
            bins_weighted_mean_squared_diff[i_bin] = np.square(
                count[i_bin] * (np.mean(bin_data[response]) - pop_mean)
            )
            # unweighted mean or response
            bins_unweighted_mean_squared_diff[i_bin] = np.square(
                (np.mean(bin_data[response]) - pop_mean)
            )

            bin_mean[i_bin] = np.mean(bin_data[response])

        MOR_Table["Bin_Mean"] = bin_mean
        MOR_Table["mean_squared_diff_weighted"] = bins_weighted_mean_squared_diff
        MOR_Table["Population_Mean"] = pop_mean

        print("w mean squared difference = ", np.mean(bins_weighted_mean_squared_diff))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=bins_edges,
                y=MOR_Table["Bin_Count"],
                name="Population",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=[bins_edges.min(), bins_edges.max()],
                y=MOR_Table["Bin_Mean"],
                mode="lines",
                line=dict(color="red", width=2),
                name="Bin Mean"
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=[bins_edges.min(), bins_edges.max()],
                y=[
                    MOR_Table["Population_Mean"][0],
                ],
                mode="lines",
                line=dict(color="red", width=2),

                name="Population Mean",
            )
        )

        fig.add_hline(
            MOR_Table["Population_Mean"][0], line_color="green"
        )

        title = f" Mean of Response vs Bin ({predictors[i]})"
        # Add figure title
        fig.update_layout(
            title_text=f"Mean of Response of {predictors[i]}"
        )

        # Set x-axis title
        fig.update_xaxes(title_text=f"Predictor - {predictors[i]} Bin")

        # Set y-axes titles
        fig.update_yaxes(title_text="Population", secondary_y=True)
        fig.update_yaxes(
            title_text=f"Response - {response}", secondary_y=False
        )

        # fig.show()
        urls = []

        if not os.path.isdir("Mean_Of_Response_1D_plots"):
            os.mkdir("Mean_Of_Response_1D_plots")
        file_path = f"Mean_Of_Response_1D_plots/{predictors[i]}-{response}-plot.html"
        urls.append(file_path)
        fig.write_html(file=file_path, include_plotlyjs="cdn")
        # Table for Single Predictor
        # print(MOR_Table)
        # print(urls[0])
        # print(response[0],predictors[i],mean_squared_diff,mean_squared_diff_weighted,urls)

        MOR_table = MOR_table.append(
            {
                'Response': response,
                'Predictor': predictors[i],
                'Mean_Squared_Error': np.mean(bins_unweighted_mean_squared_diff),
                'Mean_Squared_Error_Weighted': np.mean(bins_weighted_mean_squared_diff),
                'URL': urls[0]
            },
            ignore_index=True
        )

    MOR_table = MOR_table.sort_values("Mean_Squared_Error_Weighted", ascending=False)

    MOR_table = MOR_table.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    return MOR_table


def Brute_Force_cont_cont(df, continuous, response):
    pop_mean = df[response].values.mean()
    # print(df.head())
    Brute_Force_Table = pd.DataFrame()
    final_Brute_Force_table = pd.DataFrame(columns=[
        'Predictor_1',
        'Predictor_2',
        'Mean_Squared_Error',
        'Mean_Squared_Error_Weighted',
        'URL',
    ])
    print(type(final_Brute_Force_table))
    for a, b in combinations(continuous, 2):
        df_copy = df[[a, b, response]].copy()
        # print(df_copy.head())
        df_copy["Bin_1"] = pd.cut(df_copy[a], 10, include_lowest=True, duplicates="drop")
        df_copy["Bin_2"] = pd.cut(df_copy[b], 10, include_lowest=True, duplicates="drop")
        df_copy["Lower_Bin_1"] = (df_copy["Bin_1"].apply(lambda x: x.left).astype(float))
        df_copy["Upper_Bin_1"] = (df_copy["Bin_1"].apply(lambda x: x.right).astype(float))
        df_copy["Lower_Bin_2"] = (df_copy["Bin_2"].apply(lambda x: x.left).astype(float))
        df_copy["Upper_Bin_2"] = (df_copy["Bin_2"].apply(lambda x: x.right).astype(float))
        df_copy["Bin_Centre_1"] = (df_copy["Lower_Bin_1"] + df_copy["Upper_Bin_1"]) / 2
        df_copy["Bin_Centre_2"] = (df_copy["Lower_Bin_2"] + df_copy["Upper_Bin_2"]) / 2

        bin_mean = df_copy.groupby(by=["Bin_Centre_1", "Bin_Centre_2"]).mean().reset_index()
        # print(bin_mean)
        bin_count = df_copy.groupby(by=["Bin_Centre_1", "Bin_Centre_2"]).count().reset_index()
        Brute_Force_Table["Bin_Count"] = bin_count[response]
        Brute_Force_Table["Bin_Mean"] = bin_mean[response]

        Brute_Force_Table["Population_Mean"] = pop_mean
        Brute_Force_Table["Mean_diff"] = round(Brute_Force_Table["Bin_Mean"] - Brute_Force_Table["Population_Mean"], 6)
        Brute_Force_Table["mean_squared_diff"] = round((Brute_Force_Table["Mean_diff"]) ** 2, 6)
        Brute_Force_Table["Weight"] = (Brute_Force_Table["Bin_Count"] / df[response].count())
        Brute_Force_Table["mean_squared_diff_weighted"] = (
                Brute_Force_Table["mean_squared_diff"] * Brute_Force_Table["Weight"])
        # Brute_Force_Table = Brute_Force_Table.reset_index()
        Brute_Force_Table["mean_squared_diff"] = Brute_Force_Table["mean_squared_diff"].mean()
        mean_squared_diff = round((Brute_Force_Table["mean_squared_diff"].sum()), 6)
        mean_squared_diff_weighted = round(Brute_Force_Table["mean_squared_diff_weighted"].sum(), 6)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=df_copy["Bin_Centre_1"].values,  # Check AGAIN
                y=df_copy["Bin_Centre_2"].values,  # Check AGAIN
                z=Brute_Force_Table["Mean_diff"],
                text=Brute_Force_Table["Mean_diff"],
                texttemplate="%{text}",
            )
        )

        title = f"Brute Force Mean of Response for {a} by {b}"

        fig.update_layout(
            title=title,
            xaxis_title=f"{a}",
            yaxis_title=f"{b}",
        )

        urls = []

        if not os.path.isdir("Brute_Force_plots"):
            os.mkdir("Brute_Force_plots")
        file_path = f"Brute_Force_plots/{a}-{b}-plot.html"
        urls.append(file_path)
        fig.write_html(file=file_path, include_plotlyjs="cdn")
        # Table for Single Predictor
        # print(MOR_Table)
        # print(urls[0])
        # print(response[0],predictors[i],mean_squared_diff,mean_squared_diff_weighted,urls)

        final_Brute_Force_table = final_Brute_Force_table.append(
            {
                'Predictor_1': a,
                'Predictor_2': b,
                'Mean_Squared_Error': mean_squared_diff,
                'Mean_Squared_Error_Weighted': mean_squared_diff_weighted,
                'URL': urls[0]
            },
            ignore_index=True
        )
    final_Brute_Force_table = final_Brute_Force_table.sort_values("Mean_Squared_Error_Weighted", ascending=False)
    # final_MOR_1D_table = final_MOR_1D_table.reset_index(drop=True)
    final_Brute_Force_table = final_Brute_Force_table.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    return final_Brute_Force_table


def create_list_of_cat_con_predictors(df, predictors):
    # identify and create the list of categorical and continuous data predictors
    _continuous_predictors = []
    _categorical_predictors = []
    for predictor in predictors:
        if len(pd.unique(df[predictor])) >= 5:
            print(predictor + " variable is continuous")
            _continuous_predictors.append(predictor)
        else:
            print(predictor + " variable is categorical")
            _categorical_predictors.append(predictor)

    return _continuous_predictors, _categorical_predictors


def correlation_cont_cont(df, continuous_predictors):
    count = 0
    x_continuous_continuous = []
    y_continuous_continuous = []
    cont_cont_correlation = []
    linear_regression_plots = []
    test_cont = 0
    if len(continuous_predictors) > 0:
        for x in continuous_predictors:
            for y in continuous_predictors:
                if x != y and test_cont < 1:
                    df[x].fillna(df[x].mean(), inplace=True)
                    df[y].fillna(df[y].mean(), inplace=True)
                    pearson_r, p_val = stats.pearsonr(df[x], df[y])
                    cont_cont_correlation.append(pearson_r)
                    x_continuous_continuous.append(x)
                    y_continuous_continuous.append(y)
                    fig = px.scatter(x=df[x], y=df[y], trendline="ols")
                    fig.update_layout(
                        title="Continuous Predictor by Continuous Predictor",
                        xaxis_title="Pred1",
                        yaxis_title="Pred2",
                    )
                    # fig.show()
                    name = "linear_regression_" + str(count) + ".html"
                    path = join(dirname(abspath(__file__)), "Project_html", name)
                    fig.write_html(
                        path,
                        include_plotlyjs="cdn",
                    )
                    count += 1
                    linear_regression_plots.append("file://" + path)
        test_cont += 1

        # creating a cont-cont correlation table

        cont_cont_correlation_table = pd.DataFrame(
            columns=["cont_var1", "cont_var2", "pearson's_r", "link_plot"]
        )
        cont_cont_correlation_table["cont_var1"] = x_continuous_continuous
        cont_cont_correlation_table["cont_var2"] = y_continuous_continuous
        cont_cont_correlation_table["pearson's_r"] = cont_cont_correlation
        cont_cont_correlation_table["link_plot"] = linear_regression_plots
        cont_cont_correlation_table.style.format({"link_plot": make_clickable})
        # Put values in tables ordered DESC by correlation metric
        cont_cont_correlation_table.sort_values(
            by=["pearson's_r"], inplace=True, ascending=False
        )

        cont_cont_correlation_plot = go.Figure(
            data=go.Heatmap(
                x=cont_cont_correlation_table["cont_var1"],
                y=cont_cont_correlation_table["cont_var2"],
                z=cont_cont_correlation_table["pearson's_r"],
            )
        )
        cont_cont_correlation_plot.update_layout(
            title="continuous_continuous_correlation_plot",
            xaxis_title="cont_var1",
            yaxis_title="cont_var1",
        )
        path = join(
            dirname(abspath(__file__)),
            "Project_html",
            "cont_cont_correlation_plot.html",
        )
        cont_cont_correlation_plot.write_html(
            path,
            include_plotlyjs="cdn",
        )

    return cont_cont_correlation_plot, cont_cont_correlation_table


def main():
    categorical_predictors = []
    continuous_predictors = []

    path = join(dirname(abspath(__file__)), "Project_html")
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory if it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    df = import_data()
    response = 'HomeTeamWins'
    inspect_data(df)
    df = clean_data(df)
    print(df["local_date"])
    X_train, X_test, y_train, y_test = train_test_split(df)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # print(X_train)
    # print(y_train)
    logistic_regression(X_train, y_train, X_test, y_test)

    X_train_new = X_train.loc[:, ~X_train.columns.isin(['game_id', 'home_team', 'away_team', 'num_games', 'r_home_BB9',
                                                        'r_home_BB', 'r_away_K9', 'r_away_KBB', 'r_away_H',
                                                        'r_away_HR9', 'r_home_KBB', 'r_away_BA', 'r_home_H'])]
    X_test_new = X_test.loc[:, ~X_test.columns.isin(['game_id', 'home_team', 'away_team', 'num_games', 'r_home_BB9',
                                                     'r_home_BB', 'r_away_K9', 'r_away_KBB', 'r_away_H', 'r_away_HR9',
                                                     'r_home_KBB', 'r_away_BA', 'r_home_H'])]

    df_new = df.loc[:, ~df.columns.isin(['local_date', 'game_id', 'home_team', 'away_team', 'num_games', 'r_home_BB9',
                                         'r_home_BB', 'r_away_K9', 'r_away_KBB', 'r_away_H', 'r_away_HR9', 'r_home_KBB',
                                         'r_away_BA', 'r_home_H'])]

    df_new_pred = df.loc[:, ~df.columns.isin(['local_date', 'game_id', 'home_team', 'away_team', 'num_games',
                                              'r_home_BB9', 'r_home_BB', 'HomeTeamWins', 'r_away_K9', 'r_away_KBB',
                                              'r_away_H', 'r_away_HR9', 'r_home_KBB', 'r_away_BA', 'r_home_H'])]

    logistic_regression(X_train_new, y_train, X_test_new, y_test)

    rf_model = random_forest(X_train, y_train, X_test, y_test)
    plot_feature_importance(rf_model.feature_importances_, X_train.columns, 'RANDOM FOREST')

    decision_tree(X_train_new, y_train, X_test_new, y_test)
    naive_bayes_model(X_train_new, y_train, X_test_new, y_test)

    predictors = df_new_pred.columns
    mean_of_response_table = mean_of_response(df_new, predictors, response)

    continuous_predictors, categorical_predictors = create_list_of_cat_con_predictors(df_new_pred, predictors)
    cont_cont_correlation_plot, cont_cont_correlation_table = correlation_cont_cont(df_new_pred, continuous_predictors)

    brute_force_table = Brute_Force_cont_cont(df_new, continuous_predictors, response)

    # adding all the tables and plots on single html page
    with open("final.html", "w+") as file:
        file.write(
            mean_of_response_table.to_html(render_links=True)
            + "\n\n"
            + brute_force_table.to_html(render_links=True)
        )


if __name__ == "__main__":
    main()
