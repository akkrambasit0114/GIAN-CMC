import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("/Users/insomni_.ak/Documents/Machine Learning/GIAN_CMC/input/out.csv")  
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    target="Strength"

    # fetch targets
    y = df[target].values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)
    #print(kf)
    # fill the new kfold column
    #does this code works?? x should be df[except y]
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df,y=y)):
        #print(len(train_idx), len(val_idx))
        #print(fold, train_idx, val_idx)
        df.loc[val_idx, 'kfold'] = fold
        
    df.to_csv("/Users/insomni_.ak/Documents/Machine Learning/GIAN_CMC/input/knnimputer_db_folds.csv",index=False)
    