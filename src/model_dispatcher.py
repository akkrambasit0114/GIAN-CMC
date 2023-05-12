from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from xgboost import XGBRegressor as XGB

models = {
    "linear_model_LogReg": LinearRegression(
    ),
    "linear_model_Lasso": Lasso(
    
    ),
    "rf": ensemble.RandomForestRegressor(
    
    ),
    "xgboost": XGB(
    eta=0.05,
    gamma=0.5,
    max_depth=3
    ),
    "gboost":GBR(
    
    ),
    "adboost":ABR(
    
    ),
    "SVR":SVR(
    
    ),
    "KNR":KNR(
    
    )
 }
