import pickle
from flask import Flask, jsonify
from flask import request

app = Flask("coffee-quality-prediction")

model_file = "./models/model_v1.bin"

with open(model_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)


@app.route("/predict", methods=["POST"])
def predict():
    print("Jess")
    sample = request.get_json()
    print(sample)

    # dict = [sample.to_dict(orient='records')

    X_val = dv.transform(sample)
    y_pred = model.predict(X_val)

    result = {
        "sample_over_85_points_prediction": str(y_pred[0] == 1),
        # "score": bool(churn)
        # bool_ is coming from numpy which our service doesn't know
        # how to turn into text so we need to wrap it!
        # similary we do this for y_pred by making it a float
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
