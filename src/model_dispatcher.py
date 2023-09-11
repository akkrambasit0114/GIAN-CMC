from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ada
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import ExtraTreeRegressor as ET
from xgboost import XGBRegressor as XGB
from lightgbm import LGBMRegressor as LGBMReg
models = {
    "linear_model_LogReg": LinearRegression(
    fit_intercept= True
    ),
    "linear_model_Lasso": Lasso(
    ),
    "rf": ensemble.RandomForestRegressor(
        bootstrap=True, 
        ccp_alpha=0.0, 
        criterion='squared_error', 
        max_depth=None, 
        max_features=1.0,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        n_jobs=-1,
        oob_score=False,
        random_state=123,
        verbose=0,
        warm_start=False
    
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
    "et":ET(),

    "gboost":GBR(
    
    ),
    
    "ada":ada(
        base_estimator='deprecated',
        estimator=None,
        learning_rate=1.0,
        loss='linear',
        n_estimators=50,
        random_state=123    
    ),
    "SVR":SVR(
    
    ),
    "KNR":KNR(
    
    ),
    "LGBMR": LGBMReg(
    
    )
 }
