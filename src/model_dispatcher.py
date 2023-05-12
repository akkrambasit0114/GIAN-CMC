from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor as XGB

models = {
    "linear_model_LogReg": LinearRegression(
    ),
    "linear_model_Lasso": Lasso(
    
    ),
    "rf": ensemble.RandomForestRegressor(
    
    ),
    "xgboost": XGB(
    
    )
 }
