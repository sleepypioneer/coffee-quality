import requests

coffee_sample = {
    "contract": "two_year",
    "tenure": 1,
    "monthlycharges": 10
}

url = "http://localhost:9696/predict"

resp = requests.post(url, json=coffee_sample).json()

print(f"The coffee sample's score is predicted to be: {resp["quality_score_prediction"]}."