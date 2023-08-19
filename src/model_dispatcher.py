from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ada
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from xgboost import XGBRegressor as XGB
from lightgbm import LGBMRegressor as LGBMReg
models = {
    "linear_model_LogReg": LinearRegression(
    fit_intercept= True
    ),
    "linear_model_Lasso": Lasso(
    ),
    "rf": ensemble.RandomForestRegressor(
    
    ),
    "xgboost": XGB(
    eta=0.05,
    gamma=0.06,
    max_depth=15,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.7,
    alpha=0.1
    ),
    "gboost":GBR(
    
    ),
    "ada":ada(
    
    ),
    "SVR":SVR(
    
    ),
    "KNR":KNR(
    
    ),
    "LGBMR": LGBMReg(
    
    )
 }
