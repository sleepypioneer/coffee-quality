import numpy as np

from sklearn.model_selection import train_test_split


def handle_na_values(df):
    df["altitude_low_meters"].fillna(df["altitude_low_meters"].mean(), inplace=True)
    df["altitude_high_meters"].fillna(df["altitude_high_meters"].mean(), inplace=True)
    df["altitude_mean_meters"].fillna(df["altitude_mean_meters"].mean(), inplace=True)


    df["lot_number"].fillna("missing", inplace=True)
    df["farm_name"].fillna("missing", inplace=True)
    df["mill"].fillna("missing", inplace=True)
    df["owner"].fillna("missing", inplace=True)
    df["company"].fillna("missing", inplace=True)
    df["producer"].fillna("missing", inplace=True)
    df["ico_number"].fillna("missing", inplace=True)

    return df.dropna()


def split_data(raw_data, features):
    np.random.seed(42)
    
    df_full_train, df_test = train_test_split(raw_data, test_size=0.2, random_state=1)
    df_train, df_val  = train_test_split(df_full_train, test_size=0.25, random_state=1)

    print(f"length of training set: {len(df_train)}, validation set: {len(df_val)}, test set: {len(df_test)}")

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["total_cup_points"].values
    y_val = df_val["total_cup_points"].values
    y_test = df_test["total_cup_points"].values

    df_train = df_train[features]
    df_val = df_val[features]
    df_test = df_test[features]
    
    return df_train, df_val, df_test, y_train, y_val, y_test, df_full_train