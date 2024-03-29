
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(df, on_col, save_cols=None, seed=2020, test_size=0.2):
    if not save_cols:
        save_cols = df.columns.values

    target = df[on_col].unique()
    train, test = train_test_split(target, random_state=seed, test_size=test_size, shuffle=True)
    train_df = df[df[on_col].isin(train)]
    test_df = df[df[on_col].isin(test)]

    return train_df[save_cols], test_df[save_cols]


if __name__ == "__main__":
    from args import args

    data = pd.read_csv(args.data)
    train, test = data_split(data, "PTNM", seed=1329+args.fold, test_size=0.2)
    train, validate = data_split(train, "PTNM", seed=1329+args.fold+args.model, test_size=0.2)

    test_add_to_train = pd.DataFrame()
    test_add_to_train = pd.concat([test_add_to_train, test[(test.DSFQ == 1) & (test.TIME < 168)]], ignore_index=True)
    test_add_to_train = pd.concat([test_add_to_train, test[(test.DSFQ == 3) & (test.TIME < 504)]], ignore_index=True)
    train = pd.concat([train, test_add_to_train], ignore_index=True)
    validate = pd.concat([validate, test_add_to_train], ignore_index=True)

    # James' augmentation
    augment_data = pd.DataFrame(columns=train.columns)
    for ptnm in train.PTNM.unique():
        df = train[(train.PTNM == ptnm) & (train.TIME <= 2 * 21 * 24) & (train.TIME >= 0)]
        df["PTNM"] = df["PTNM"] + 0.1
        augment_data = pd.concat([augment_data, df], ignore_index=True)

        df = train[(train.PTNM == ptnm) & (train.TIME <= 3 * 21 * 24) & (train.TIME >= 0)]
        df["PTNM"] = df["PTNM"] + 0.2
        augment_data = pd.concat([augment_data, df], ignore_index=True)

        df = train[(train.PTNM == ptnm) & (train.TIME <= 4 * 21 * 24) & (train.TIME >= 0)]
        df["PTNM"] = df["PTNM"] + 0.3
        augment_data = pd.concat([augment_data, df], ignore_index=True)

    train = pd.concat([train, augment_data], ignore_index=True).reset_index(drop=True)


    train.to_csv("/Users/rishabhgoel/Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/results/train.csv", index=False) 
    validate.to_csv("/Users/rishabhgoel/Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/results/validate.csv", index=False)
    test.to_csv("/Users/rishabhgoel/Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/results/test.csv", index=False)
