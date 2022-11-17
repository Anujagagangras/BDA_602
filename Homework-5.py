import sys
import numpy as np
import pandas
import statsmodels.api
import sqlalchemy
import pandas as pd
import plotly.express as px

categorical_predictors = []
continuous_predictors = []
type_of_response = []
type_of_predictor = []
stats_table = pd.DataFrame()
def main():
    db_user = "root"

    db_pass = "Shekharr1986#"  # pragma: allowlist secret

    db_host = "localhost"

    db_database = "baseball"

    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """

        SELECT *
        FROM Result_table
            """

    df = pandas.read_sql_query(query, sql_engine)

    print(df.head())
    my_list = list(df)

    print(my_list)
    # response variable (user can change this)
    response = "Home_Team_Wins"

    # drop rows which contain missing values
    df.dropna(axis=0)

    # create dataframe of only predictors
    df_predictors = df.loc[:, df.columns != response]

    # create a list of column names
    predictors = df_predictors.columns

    # empty tables of dataframes for storing correlation values of all the combinations
    cat_cat_correlation_table = pd.DataFrame()
    cat_cont_correlation_table = pd.DataFrame()
    cont_cont_correlation_table = pd.DataFrame()

    # identify and create the list of categorical and continuous data predictors
    for predictor in predictors:
        if len(pd.unique(df[predictor])) >= 3:
            print(predictor + " variable is continous")

        else:
            print(predictor + " variable is categorical")
            categorical_predictors.append(predictor)


    # making a list of predictor types
    for predictor in predictors:
        if predictor in continuous_predictors:
            type_of_predictor.append("continuous_predictor")
        else:
            type_of_predictor.append("categorical_predictor")

    # determining whether response variable is categorical or continuous
    # assign response types
    if len(pd.unique(df[response])) <= 5:
        print("Response variable is boolean")
        type_of_response.append("categorical_response")
        is_bool = True
    else:
        type_of_response.append("continuous_response")
        print("Response variable is continuous")
        is_bool = False



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

if __name__ == "__main__":
    sys.exit(main())