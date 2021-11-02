import requests

coffee_sample = {
    "moisture": 0.12,
    "quakers": 0.0,
    "category_two_defects": 15,
    "species": "Arabica",
    "owner": "sanjava coffee",
    "farm_name": "various",
    "company": "pt. shriya artha nusantara",
    "region": "sapan toraja",
    "producer": "vary farm",
    "in_country_partner": "Specialty Coffee Association of Indonesia",
    "harvest_year": 2017,
    "owner_1": "SanJava Coffee",
    "variety": "Sulawesi",
}

url = "https://coffee-quality-prediction.herokuapp.com/predict"

resp = requests.post(url, json=coffee_sample).json()

print(
    "The coffee sample is predicted to have a total cup score"
    f" over 85: {resp['sample_over_85_points_prediction']}."
)
