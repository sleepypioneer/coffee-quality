import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

output_file = "./models/model_v1.bin"

features = [
    "moisture",
    "quakers",
    "category_two_defects",
    "species",
    "owner",
    "farm_name",
    "company",
    "region",
    "producer",
    "in_country_partner",
    "harvest_year",
    "owner_1",
    "variety",
]


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


def convert_bag_weight(weight_string):
    return 0


def data_prep(df):
    df.columns = df.columns.str.lower().str.replace(".", "_")
    df = handle_na_values(df)
    df["bag_weight"] = df["bag_weight"].apply(
        lambda weight_str: convert_bag_weight(weight_str)
    )
    return df


def total_points_over_85(y):
    y = [1 if x > 85 else 0 for x in y]
    return y


def split_data(raw_data, features):
    np.random.seed(42)

    df_full_train, df_test = train_test_split(raw_data, test_size=0.2, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train["total_cup_points"].values
    y_test = df_test["total_cup_points"].values

    df_full_train = df_full_train[features]
    df_test = df_test[features]

    return df_full_train, y_full_train, df_test, y_test


def train_final_model(df, y_train, c=1, max_iter=100, random_state=2):
    dicts = df.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(
        solver="liblinear", C=c, max_iter=max_iter, random_state=random_state
    )
    model.fit(X_train, y_train)

    return dv, model


def validate_model(df_val, y_val, dv, model):
    val_dicts = df_val.to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_preds = model.predict(X_val)
    return y_preds


def model_evaluation(y_val, y_pred):
    precision = precision_score(y_val, y_pred, average="binary")
    f1 = f1_score(y_val, y_pred, average="binary")
    recall = recall_score(y_val, y_pred, average="binary")
    accuracy = (y_pred == y_val).mean()
    auc = roc_auc_score(y_val, y_pred)
    rsme = mean_squared_error(y_val, y_pred, squared=False)

    return accuracy, auc, rsme, precision, f1, recall


def print_model_evaluation(y_val, y_pred):
    accuracy, auc, rsme, precision, f1, recall = model_evaluation(y_val, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Roc Auc: {auc:.2f}")
    print(f"Rsme: {rsme:.2f}")

    print("Precision: %.3f" % precision)
    print("F-Measure: %.3f" % f1)
    print("Recall: %.3f" % recall)


if __name__ == "__main__":
    df = pd.read_csv("./data/merged_data_cleaned.csv", index_col=0)
    df = data_prep(df)

    df_full_train, y_full_train, df_test, y_test = split_data(df, features)

    # this converts the score to a binary classification
    # for if the total score is above 85 points or not
    y_full_train = total_points_over_85(y_full_train)
    y_test = total_points_over_85(y_test)

    # parameters
    c = 1
    max_iter = 100
    random_state = 2

    dv, model = train_final_model(
        df_full_train, y_full_train, c=c, max_iter=max_iter, random_state=random_state
    )

    y_pred = validate_model(df_test, y_test, dv, model)

    print(
        "Final model is: Logistic Regression, with parameters: "
        f"c={c}, max_iter={max_iter}, random_state={random_state}"
    )
    print_model_evaluation(y_test, y_pred)

    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print("model saved")
