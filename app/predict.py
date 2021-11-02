import pickle
from flask import Flask, jsonify
from flask import request

app = Flask("coffee-quality-prediction")

model_file = "./models/model_v1.bin"

with open(model_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)


@app.route("/predict", methods=["POST"])
def predict():
    sample = request.get_json()

    X_val = dv.transform(sample)
    y_pred = model.predict(X_val)

    result = {
        "sample_over_85_points_prediction": str(y_pred[0] == 1),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
