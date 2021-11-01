import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(
        features,
        df_train,
        y_train,
        df_val,
        y_val,
        C=1.0,
        feature_to_remove=""
    ):
    if feature_to_remove:
        X_train_without_room_type = df_train[feature_to_remove]
        X_val_without_room_type = df_val[feature_to_remove]
        sub_features = np.array([x for x in features if x != feature_to_remove])
    else:
        sub_features = features
    train_dicts = df_train[sub_features].to_dict(orient='records')
    
    train_dict = df_train[sub_features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    model = model = LogisticRegression(solver='liblinear', C=C, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[sub_features].to_dict(orient='records')

    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    predict_positive = (y_pred == 1)
    predict_negative = (y_pred == 0)
    
#     print(actual_positive, actual_negative, predict_positive, predict_negative)
    
    true_positive = (predict_positive & actual_positive).sum()
    true_negative = (predict_negative & actual_negative).sum()
    false_positive = (predict_positive & actual_negative).sum()
    false_negative = (predict_negative & actual_positive).sum()
    
#     print(np.array([
#         [true_negative, false_positive],
#         [false_negative, true_positive]
#     ]))
    
    precision = round(true_positive / (true_positive + false_positive), 4)
    recall = round(true_positive / (true_positive + false_negative), 4)
    
    return (feature_to_remove, C, (y_val == y_pred).mean(), precision, recall)