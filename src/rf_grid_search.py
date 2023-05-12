# rf_grid_search.py
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from xgboost import XGBRegressor as XGB

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("/Users/insomni_.ak/Documents/Machine Learning/GIAN_CMC/input/knnimputer_db.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("Strength", axis=1).values
    # and the targets
    y = df.Strength.values

    # define the model here
    # i am using random forest with n_jobs=-1
    # n_jobs=-1 => use all cores
    regressor = XGB(n_jobs=-1)

    # define a grid of parameters
    # this can be a dictionary or a list of
    # dictionaries
    param_grid={
        "eta": [0.01, 0.015, 0.025, 0.05, 0.1],
        "gamma": [0.05,0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "max_depth":[3, 5, 7, 9, 12, 15, 17, 25],
        # "min_child_weight":[1, 3, 5, 7],
        # "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        # "colsample_bytree":[0.6, 0.7, 0,8, 0.9, 1.0],
        "lambda":[0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 1.0],
        # "alpha":[0, 0.1, 0.5, 1.0]
    }

    # initilize grid searchd
    # estimator is the model that we have defined
    # param_grid is the grid of parameters
    # we use accuracy as our metric. you can define your own
    # higher value of verbose implies a lot of details are printed
    # cv=5 means that we are using 5 fold cv (not stratified)
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="r2",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set: ")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
