import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from evaluate import validate_model, model_evaluation


def train_logistic_regression(df, y_train, c=1.0, max_iter=1000, random_state=42):
    dicts = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(solver='liblinear', C=c, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    return dv, model


def train_decision_tree(df, y_train):
    train_dicts = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X_train, y_train)
    
    return dv, model
    

def train_random_forest(df, y_train):
    train_dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return dv, model


def feature_importance(
        features,
        df_train,
        y_train,
        df_val,
        y_val,
        feature_to_remove=""
    ):
    if feature_to_remove:
        sub_features = np.array([x for x in features if x != feature_to_remove])
    else:
        sub_features = features
    
    dv, model = train_logistic_regression(df_train[sub_features], y_train)

    y_pred = validate_model(df_val, y_val, dv, model)

    accuracy, auc, rsme, precision, f1, recall =  model_evaluation(y_val, y_pred)

    
    return (feature_to_remove, accuracy, auc, rsme, precision, f1, recall)


def check_feature_importance(
        features,
        df_train,
        y_train,
        df_val,
        y_val
    ):
    scores = []
    for feature in features:
        scores.append(feature_importance(
            features,
            df_train,
            y_train,
            df_val,
            y_val,
            feature
        ))

    cols = ["feature_removed", "accuracy", "auc", "rsme", "precision", "f1_score", "recall"]
    return pd.DataFrame(scores, columns=cols)