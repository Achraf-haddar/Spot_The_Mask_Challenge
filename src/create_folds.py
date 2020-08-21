from sklearn import model_selection
import pandas as pd
import numpy as np

csv_path = "../input/train_labels.csv"
df = pd.read_csv(csv_path)
y = df["target"]
df["kfold"] = -1
kf = model_selection.StratifiedKFold(n_splits=5)
for fold_, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
    df.loc[val_, "kfold"] = fold_
print(df.head())

df.to_csv("input/train_folds.csv")