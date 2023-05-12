# src/train.py

import os
import config
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import argparse
import model_dispatcher

def run(fold, model):
    # read the training data with folds
    #f = open("/mnt/c/Users/Basit Akram/Documents/new-project/input/train_folds.csv")
    df = pd.read_csv(config.TRAINING_FILE)
    df = pd.get_dummies(df)

    # training data is where kfold is not euqal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values
    # target is label column in the dataframe
    x_train = df_train.drop("target", axis=1).values
    y_train = df_train.target.values

    # similarly, for validation, we have
    x_valid = df_valid.drop("target", axis=1).values
    y_valid = df_valid.target.values

    # fetch the model from model_dispatcher
    print(model)
    clf = model_dispatcher.models[model]
    # fit the model on the training data
    clf.fit(x_train, y_train) #####

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & predict accuracy
    accuracy = metrics. accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT,f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    # initialize ArguementParser class of argparse
    parser = argparse.ArgumentParser()
    
    # add the different arguments you need and their type
    # currently, we only need fold
    
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model", 
        type=str
    )
    # read the argument from the command line
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    run(fold=args.fold, 
        model = args.model
    )