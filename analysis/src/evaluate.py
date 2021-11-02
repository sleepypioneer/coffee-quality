from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
)


def validate_model(df_val, y_val, dv, model):
    val_dicts = df_val.to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_preds = model.predict(X_val)
    return y_preds


def model_evaluation(y_val, y_pred):
    accuracy = (y_pred == y_val).mean()
    auc = roc_auc_score(y_val, y_pred)
    rsme = mean_squared_error(y_val, y_pred, squared=False)

    return accuracy, auc, rsme


def print_model_evaluation(y_val, y_pred):
    accuracy, auc, rsme = model_evaluation(y_val, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Roc Auc: {auc:.2f}")
    print(f"Rsme: {rsme:.2f}")
