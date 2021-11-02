from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, mean_squared_error


def validate_model(df_val, y_val, dv, model):
    val_dicts = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_preds = model.predict(X_val)
    return y_preds


def model_evaluation(y_val, y_pred):
    precision = precision_score(y_val, y_pred, average='binary')
    f1 = f1_score(y_val, y_pred, average='binary')
    recall = recall_score(y_val, y_pred, average='binary')
    accuracy =  (y_pred == y_val).mean()
    auc = roc_auc_score(y_val, y_pred)
    rsme = mean_squared_error(y_val, y_pred, squared=False)

    return accuracy, auc, rsme, precision, f1, recall


def print_model_evaluation(y_val, y_pred):
    accuracy, auc, rsme, precision, f1, recall = \
        model_evaluation(y_val, y_pred)
        
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Roc Auc: {auc:.2f}")
    print(f"Rsme: {rsme:.2f}")

    print('Precision: %.3f' % precision)
    print('F-Measure: %.3f' % f1)
    print('Recall: %.3f' % recall)