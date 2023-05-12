# python3 train.py --fold 0 --model linear_model_LogReg
# python3 train.py --fold 1 --model linear_model_LogReg
# python3 train.py --fold 2 --model linear_model_LogReg
# python3 train.py --fold 3 --model linear_model_LogReg
# python3 train.py --fold 4 --model linear_model_LogReg
# python3 train.py --fold 0 --model linear_model_Lasso
# python3 train.py --fold 1 --model linear_model_Lasso
# python3 train.py --fold 2 --model linear_model_Lasso
# python3 train.py --fold 3 --model linear_model_Lasso
# python3 train.py --fold 4 --model linear_model_Lasso
# python3 train.py --fold 0 --model rf
# python3 train.py --fold 1 --model rf
# python3 train.py --fold 2 --model rf
# python3 train.py --fold 3 --model rf
# python3 train.py --fold 4 --model rf
# python3 train.py --fold 0 --model xgboost
# python3 train.py --fold 1 --model xgboost
# python3 train.py --fold 2 --model xgboost
# python3 train.py --fold 3 --model xgboost
# python3 train.py --fold 4 --model xgboost
# python3 train.py --fold 0 --model gboost
# python3 train.py --fold 1 --model gboost
# python3 train.py --fold 2 --model gboost
# python3 train.py --fold 3 --model gboost
# python3 train.py --fold 4 --model gboost
# python3 train.py --fold 0 --model adboost
# python3 train.py --fold 1 --model adboost
# python3 train.py --fold 2 --model adboost
# python3 train.py --fold 3 --model adboost
# python3 train.py --fold 4 --model adboost
# python3 train.py --fold 0 --model SVR
# python3 train.py --fold 1 --model SVR
# python3 train.py --fold 2 --model SVR
# python3 train.py --fold 3 --model SVR
# python3 train.py --fold 4 --model SVR
# python3 train.py --fold 0 --model KNR
# python3 train.py --fold 1 --model KNR
# python3 train.py --fold 2 --model KNR
# python3 train.py --fold 3 --model KNR
# python3 train.py --fold 4 --model KNR


# max=10
# for (( i=1; i <= $max; ++i ))
# do
#     python3 create_folds.py
#     python3 train.py --fold 0 --model adboost
#     python3 train.py --fold 1 --model adboost
#     python3 train.py --fold 2 --model adboost
#     python3 train.py --fold 3 --model adboost
#     python3 train.py --fold 4 --model adboost
# done
