import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api
from sklearn.ensemble import RandomForestClassifier

html_table = pd.DataFrame()
stats_table = pd.DataFrame()

# Step 2.Determine if response is continuous or boolean


def check_response_type(df, response):
    if len(pd.unique(df[response])) == 2:
        print("Response variable is boolean")
        _isBool = True
    else:
        print("Response variable is continuous")
        _isBool = False
    return _isBool


# step 3. Determine if the predictor is cat/cont
def check_predictor_type(df, predictor):
    if len(pd.unique(df[predictor])) >= 15 and df[predictor].dtype == float:
        _isCat = False
        print(predictor + " variable is continous")
    else:
        print(predictor + " variable is categorical")
        _isCat = False

    return _isCat


# step 4. Automatically generate the necessary plot(s) to inspect it.
# if response variable is categorical and predictor is categorical.
def plot_bool_response_cat_predictor(df, predictor, response):
    fig = px.density_heatmap(df, x=predictor, y=response)

    fig.show()


# if response variable is categorical and predictor is continous.
def plot_bool_response_con_predictor(df, predictor, response, stats_table):
    group_labels = ["Response = 0", "Response = 1"]

    df = df.dropna()

    group1 = df[df[response] == 0]
    group2 = df[df[response] == 1]

    hist_data = [group1[predictor], group2[predictor]]

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=10)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    count, bins_edges = np.histogram(df[predictor], bins=4)

    # print(count)
    # print(bins_edges)
    bins_weighted_mean_diff = [0 for i in range(len(bins_edges))]
    bins_unweighted_mean_diff = [0 for i in range(len(bins_edges))]
    print()

    pop_mean = np.mean(df[predictor])
    print("population mean for predictor ", predictor, " = ", pop_mean)

    for i in range(len(bins_edges) - 1):
        bin_data = df[
            (bins_edges[i + 1] >= df[predictor]) & (bins_edges[i] < df[predictor])
        ]
        # step 5.Difference with mean of response along with its plot (weighted and unweighted)
        # print(bin_data[predictor])
        # print()
        # weighted mean or response
        bins_weighted_mean_diff[i] = np.square(
            count[i] * (np.mean(bin_data[predictor]) - pop_mean)
        )
        # unweighted mean or response
        bins_unweighted_mean_diff[i] = np.square(
            (np.mean(bin_data[predictor]) - pop_mean)
        )
        print(
            "for predictor ",
            predictor,
            "squared diff of mean for bin ",
            i,
            " is ",
            bins_weighted_mean_diff[i],
        )

    print("w mean squared difference = ", np.mean(bins_weighted_mean_diff))
    stats_table["w_mean_diff"][predictor] = np.mean(bins_weighted_mean_diff)
    stats_table["uw_mean_diff"][predictor] = np.mean(bins_unweighted_mean_diff)


# if response variable is continous and predictor is continous.


def plot_cont_response_cont_predictor(df, predictor, response):
    x = df[predictor]
    y = df[response]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()

    return


# step 6. p-value & t-score (continuous predictors only) along with it's plot
# logistic regression will work if response variable it categorical
def logistic_regression(df, predictor_name, response, stats_table):
    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    df = pd.DataFrame({predictor_name: x, response: y})
    pd.set_option("mode.use_inf_as_na", True)
    df = df.dropna()
    x = df[predictor_name]
    y = df[response].map(int)

    predictor = statsmodels.api.add_constant(x)
    logistic_regression_model = statsmodels.api.Logit(
        np.asarray(y), np.asarray(predictor)
    )
    logistic_regression_model_fitted = logistic_regression_model.fit()
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:6e}".format(logistic_regression_model_fitted.pvalues[1])
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",
    )
    fig.show()
    stats_table["tvalue"][predictor_name] = t_value
    stats_table["pvalue"][predictor_name] = p_value
    print(stats_table)


# step 6. p-value & t-score (continuous predictors only) along with it's plot
# linear regression will work if response variable it continous
def linear_regression(df, predictor_name, response, stats_table):
    x = df[predictor_name]
    y = df[response]

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
    stats_table["tvalue"][predictor_name] = t_value
    stats_table["pvalue"][predictor_name] = p_value
    print(stats_table)


def main():
    # import data

    global stats_table
    # step 1. Given titanic pandas dataframe which contains both response and predictor columns

    dataframe = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )

    predictors = ["pclass", "sex", "age", "fare", "embarked"]
    response = "survived"

    dataframe = dataframe[["pclass", "sex", "age", "fare", "embarked", "survived"]]

    # cleaning the dataset
    dataframe.dropna()
    dataframe.reset_index(drop=True)
    dataframe["age"].fillna(dataframe["age"].mean(), inplace=True)
    dataframe["sex"].replace(["male", "female"], [0, 1], inplace=True)
    dataframe["embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)
    dataframe["embarked"].fillna(dataframe["embarked"].mean(), inplace=True)

    html_table = pd.DataFrame(
        [0.0, 0.0, 0.0, 0.0, 0.0], index=predictors, columns=["dummy"]
    )

    stats_table = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=predictors,
        columns=["tvalue", "pvalue", "uw_mean_diff", "w_mean_diff"],
    )

    # print(html_table)

    is_bool = check_response_type(dataframe, response)

    for predictor in predictors:

        is_cat = check_predictor_type(dataframe, predictor)

        # plot responses
        if is_cat is True and is_bool is True:
            plot_bool_response_cat_predictor(dataframe, predictor, response)
        if is_cat is False and is_bool is True:
            plot_bool_response_con_predictor(
                dataframe, predictor, response, stats_table
            )
        if is_cat is True and is_bool is False:
            plot_cont_response_cont_predictor(dataframe, predictor, response)

        # perform regression to get t-value ans p-value
        if is_bool is True and is_cat is False:
            logistic_regression(dataframe, predictor, response, stats_table)
        else:
            linear_regression(dataframe, predictor, response, stats_table)

    # separate data and target for random forest
    # step 7. Random Forest Variable importance ranking (continuous predictors only)
    X = dataframe.drop(response, axis=1)
    y = dataframe[response]

    rf = RandomForestClassifier()
    rf.fit(X, y)

    feature_importance = pd.DataFrame(
        rf.feature_importances_, index=X.columns, columns=["importance"]
    ).sort_values("importance", ascending=True)
    print(feature_importance)

    # step 8. Generate a table with all the variables and their rankings
    html_table = pd.concat([html_table, feature_importance, stats_table], axis=1)
    html = html_table.to_html()

    # write html to file
    text_file = open("hw4_statistics.html", "w")
    text_file.write(html)
    text_file.close()


if __name__ == "__main__":
    main()
