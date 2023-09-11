
# python3 train.py --fold 0 --model linear_model_LogReg
# python3 train.py --fold 1 --model linear_model_LogReg
# python3 train.py --fold 2 --model linear_model_LogReg
# python3 train.py --fold 3 --model linear_model_LogReg
# python3 train.py --fold 4 --model linear_model_LogReg
python3 create_folds.py
python3 train.py --fold 0 --model rf
python3 train.py --fold 1 --model rf
python3 train.py --fold 2 --model rf
python3 train.py --fold 3 --model rf
python3 train.py --fold 4 --model rf
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
# python3 train.py --fold 0 --model LGBMR
# python3 train.py --fold 1 --model LGBMR
# python3 train.py --fold 2 --model LGBMR
# python3 train.py --fold 3 --model LGBMR
# python3 train.py --fold 4 --model LGBMR

# max=10
# for (( i=1; i <= $max; ++i ))
# do
#     python3 create_folds.py
#     v1=`python3 train.py --fold 0 --model rf`
#     v2=`python3 train.py --fold 1 --model rf`
#     v3=`python3 train.py --fold 2 --model rf`
#     v4=`python3 train.py --fold 3 --model rf`
#     v5=`python3 train.py --fold 4 --model rf`
#     avg=$((v1+v2+v3+v4+v5))
#     echo $avg
# done


# python3 train.py --fold 0 --model xgboost
# python3 train.py --fold 1 --model xgboost
# python3 train.py --fold 2 --model xgboost
# python3 train.py --fold 3 --model xgboost
# python3 train.py --fold 4 --model xgboost


# max=1
# for (( i=1; i <= $max; ++i ))
# do
#     sum=0
#     python3 create_folds.py
#     max2=5
#     for (( j=1; j < $max2; ++j ))
#     do
#         python3 train.py --fold $j --model xgboost
#     done
#     v1=`python3 train.py --fold 0 --model xgboost`
#     v2=`python3 train.py --fold 1 --model xgboost`
#     v3=`python3 train.py --fold 2 --model xgboost`
#     v4=`python3 train.py --fold 3 --model xgboost`
#     v5=`python3 train.py --fold 4 --model xgboost`
#     avg=$((v1+v2+v3+v4+v5))
# done

# echo $sum


# max=10
# for (( i=1; i <= $max; ++i ))
# do
#     sum=0
#     python3 create_folds.py
#     max2=5
#     summ=0
#     for (( j=0; j < $max2; ++j ))
#     do
#         v=$(python3 train.py --fold $j --model xgboost)
#         summ=$(echo "$summ + $v" | bc)
#     done
#     avg=$(echo "$summ / 5" | bc -l)
#     #echo $avg
#     sum=$(echo "$sum + $avg" | bc)   
# done
# echo "$sum / $max"