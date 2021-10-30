from flask import Flask
from flask import request

app = Flask('coffee-quality-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    sample = request.get_json()

    X = dv.transform([sample])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        'quality_score_probability' : float(y_pred),
        'score': bool(churn)
        # bool_ is coming from numpy which our service doesn't know how to turn into text so we need to wrap it!
        # similary we do this for y_pred by making it a float
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)