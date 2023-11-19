# from tuneta.optimize import _weighted_spearman
import re

import dcor
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tabulate import tabulate


def col_name(function, study_best_params):
    """
    Create consistent column names given string function and params
    :param function:  Function represented as string
    :param study_best_params:  Params for function
    :return:
    """

    # Optuna string of indicator
    function_name = function.split("(")[0].replace(".", "_")

    # Optuna string of parameters
    params = (
        re.sub("[^0-9a-zA-Z_:,]", "", str(study_best_params))
        .replace(",", "_")
        .replace(":", "_")
    )

    # Concatenate name and params to define
    col = f"{function_name}_{params}"

    # Remove any trailing underscores
    col = re.sub(r"_$", "", col)
    return col
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression,RidgeCV,ElasticNetCV
def distance_correlation(a, b,corr=False):
    
    #try:
    #    c = spearmanr(a,b)
    #    c = abs(c.correlation)
    #except Exception as e:
    #    c = 0
    if not corr:
        # try:
        #     c = RidgeCV().fit(b.reshape(-1, 1),a.reshape(-1, 1)).coef_.flatten()[0]
        # except:
        #     c = 0
        # try:
        #     c = spearmanr(a,b)
        #     try:
        #         c = abs(c.correlation)
        #     except:
        #         c = abs(c.statistic)
        # except Exception as e:
        #     c = 0
        c = dcor.distance_correlation(a, b)
    else:
        c = dcor.distance_correlation(a, b)
    # if c > 0.5:
    #     print('e')
    return c


def remove_consecutive_duplicates_and_nans(s):
    shifted = s.astype(object).shift(-1, fill_value=object())
    return s.loc[(shifted != s) & ~(shifted.isna() & s.isna())]


# import seaborn as sns
# import matplotlib.pyplot as plt


def gen_plot(indicators, title):
    data = pd.DataFrame()
    for fitted in indicators.fitted:
        fitted.fitness = []
        fitted.length = []
        for trial in fitted.study.trials:
            print(trial)
            fitted.fitness.append(trial.value)
            fitted.length.append(trial.params["length"])
        fitted.fitness = pd.Series(fitted.fitness, name="Correlation")
        fitted.length = pd.Series(fitted.length, name="Length")
        fitted.data = pd.DataFrame([fitted.fitness, fitted.length]).T
        fitted.fn = fitted.function.split("(")[0]
        fitted.data["Indicator"] = fitted.fn
        data = pd.concat([data, fitted.data])
        fitted.x = fitted.study.best_params["length"]
        fitted.y = fitted.study.best_value

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Length", y="Correlation", data=data, hue="Indicator")
    plt.title(title)
    for fit in indicators.fitted:
        plt.vlines(x=fit.x, ymin=0, ymax=fit.y, linestyles="dotted")
