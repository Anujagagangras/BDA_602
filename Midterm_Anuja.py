import sys
import os
import warnings
import plotly.express as px
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import scipy.stats as stats
from os.path import abspath, dirname, join

# to create links of the plots
def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


categorical_predictors = []
continuous_predictors = []
type_of_response = []
type_of_predictor = []
x_categorical_categorical = []
y_categorical_categorical = []
cat_cat_correlation = []
x_categorical_continuous = []
y_categorical_continuous = []
cat_cont_correlation = []
x_continuous_continuous = []
y_continuous_continuous = []
cont_cont_correlation = []

all_buckets = []


def main():
    path = join(dirname(abspath(__file__)), "midterm_html")
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory if it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    # fetch data from website
    input_file = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

    # convert csv data into pandas dataframe
    df = pd.read_csv(input_file)

    # response variable (user can change this)
    response = "survived"

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
        if len(pd.unique(df[predictor])) >= 10 and df[predictor].dtype == float:
            _isCat = False
            print(predictor + " variable is continuous")
            continuous_predictors.append(predictor)
            df[predictor].fillna(df[predictor].mean(), inplace=True)
        else:
            print(predictor + " variable is categorical")
            _isCat = True
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
        _isBool = True
        type_of_response.append("categorical_response")
    else:
        type_of_response.append("continuous_response")
        print("Response variable is continuous")

    # code for categorical_categorical correlation
    count = 0
    heatmap_plots_links = []
    if len(categorical_predictors) > 0:
        for x in categorical_predictors:
            for y in categorical_predictors:
                if x != y:
                    corr = cat_cat_correlationelation(df[x], df[y], bias_correction=True, tschuprow=False, )
                    # print(corr)
                    cat_cat_correlation.append(corr)

                    heatmap_plot = px.density_heatmap(df, x=x, y=y)

                    heatmap_plot.update_layout(
                        title="cat_cat_correlation_plot",
                        xaxis_title=str(x),
                        yaxis_title=str(y),
                    )
                    name = "heatmap_plot_" + str(count) + ".html"
                    path = join(dirname(abspath(__file__)), "midterm_html", name)

                    count += 1
                    heatmap_plot.write_html(path, include_plotlyjs="cdn", )
                    heatmap_plots_links.append("file://" + path)
                    x_categorical_categorical.append(x)
                    y_categorical_categorical.append(y)

        # creating a cat-cat correlation tableBDA_602

        cat_cat_correlation_table = pd.DataFrame(
            columns=["cat_var1", "cat_var2", "Absolute_value_correlation", "link_plot"]
        )
        cat_cat_correlation_table["cat_var1"] = x_categorical_categorical
        cat_cat_correlation_table["cat_var2"] = y_categorical_categorical
        cat_cat_correlation_table["Absolute_value_correlation"] = cat_cat_correlation
        cat_cat_correlation_table["link_plot"] = heatmap_plots_links

        cat_cat_correlation_table.style.format({"link_plot": make_clickable})
        # Put values in tables ordered DESC by correlation metric
        cat_cat_correlation_table.sort_values(by=["Absolute_value_correlation"], inplace=True, ascending=False)

        # creating cat-cat correlation heatmap
        cat_cat_correlation_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cat_correlation_table["cat_var1"],
                y=cat_cat_correlation_table["cat_var2"],
                z=cat_cat_correlation_table["Absolute_value_correlation"],
            )
        )
        cat_cat_correlation_plot.update_layout(
            title="categorical_categoical_correlation_plot",
            xaxis_title="cat_var1",
            yaxis_title="cat_var2",
        )
        path = join(dirname(abspath(__file__)), "midterm_html", "cat_cat_correlation_plot.html")
        cat_cat_correlation_plot.write_html(path, include_plotlyjs="cdn", )

    # code for categorical_continuous correlation
    count = 0
    violin_plots_links = []
    if len(categorical_predictors) > 0 and len(continuous_predictors) > 0:
        for x in continuous_predictors:
            for y in categorical_predictors:
                corr = cat_cont_correlationelation_ratio(
                    np.asarray(df[y].unique()), np.asarray(df[x])
                )
                cat_cont_correlation.append(corr)
                violin_plot = px.violin(x=df[x], y=df[y])

                name = "violin_plot_" + str(count) + ".html"
                path = join(dirname(abspath(__file__)), "midterm_html", name)
                count += 1

                x_categorical_continuous.append(x)
                y_categorical_continuous.append(y)
                violin_plots_links.append("file://" + path)

                # creating a cat-cat correlation table
                violin_plot.write_html(path, include_plotlyjs="cdn", )

        cat_cont_correlation_table = pd.DataFrame(
            columns=["cont_var", "cat_var", "Absolute_value_correlation", "link_plot"]
        )
        cat_cont_correlation_table["cont_var"] = x_categorical_continuous
        cat_cont_correlation_table["cat_var"] = y_categorical_continuous
        cat_cont_correlation_table["Absolute_value_correlation"] = cat_cont_correlation
        cat_cont_correlation_table["link_plot"] = violin_plots_links

        cat_cat_correlation_table.style.format({"link_plot": make_clickable})

        print(cat_cont_correlation_table)

        # creating cat-cat correlation heatmap

        cat_cont_correlation_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cont_correlation_table["cont_var"],
                y=cat_cont_correlation_table["cat_var"],
                z=cat_cont_correlation_table["Absolute_value_correlation"],
            )
        )
        cat_cont_correlation_plot.update_layout(
            title="categoical_continuous_correlation_plot",
            xaxis_title="cat_var",
            yaxis_title="cont_var",
        )
        path = join(dirname(abspath(__file__)), "midterm_html", "cat_cont_correlation_plot.html")
        cat_cont_correlation_plot.write_html(path, include_plotlyjs="cdn", )

    # code for continuous continuous predictors
    count = 0
    linear_regression_plots = []
    if len(continuous_predictors) > 0:
        for x in continuous_predictors:
            for y in continuous_predictors:
                if x != y:
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
                    path = join(dirname(abspath(__file__)), "midterm_html", name)
                    fig.write_html(path, include_plotlyjs="cdn", )
                    count += 1
                    linear_regression_plots.append("file://" + path)

        # creating a cont-cont correlation table

        cont_cont_correlation_table = pd.DataFrame(
            columns=["cont_var1", "cont_var2", "pearson's_r", "link_plot"]
        )
        cont_cont_correlation_table["cont_var1"] = x_continuous_continuous
        cont_cont_correlation_table["cont_var2"] = y_continuous_continuous
        cont_cont_correlation_table["pearson's_r"] = cont_cont_correlation
        cont_cont_correlation_table["link_plot"] = linear_regression_plots
        cont_cont_correlation_table.style.format({"link_plot": make_clickable})

        print(cont_cont_correlation_table)

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
        path = join(dirname(abspath(__file__)), "midterm_html", "cont_cont_correlation_plot.html")
        cont_cont_correlation_plot.write_html(path, include_plotlyjs="cdn", )


    # brute-force table
    # cat-cat

    # cont-cat

    # cont-cont


    # adding all the tables and plots on single html page
    with open("final.html", 'w+') as file:
        file.write(
            cat_cat_correlation_table.to_html(render_links=True) + "\n\n"
            + cat_cat_correlation_plot.to_html()
            + cat_cont_correlation_table.to_html(render_links=True) + "\n\n"
            + cat_cont_correlation_plot.to_html()
            + cont_cont_correlation_table.to_html(render_links=True) + "\n\n"
            + cont_cont_correlation_plot.to_html()
        )


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


# code for cat_cat correlation taken from lecture notes

def cat_cat_correlationelation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


# code for cat_cont correlation taken from lecture notes

def cat_cont_correlationelation_ratio(categories, values):
    #     Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    #     SOURCE:
    #     1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    #     :param categories: Numpy array of categories
    #     :param values: Numpy array of values
    #     :return: correlation
    #

    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(
            n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


if __name__ == "__main__":
    main()
