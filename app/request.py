import requests

coffee_sample = {
    "moisture":0.12,
    "quakers":0.0,
    "category_two_defects":15,
    "species":"Arabica",
    "owner":"sanjava coffee",
    "farm_name":"various",
    "company":"pt. shriya artha nusantara",
    "region":"sapan toraja",
    "producer":"vary farm",
    "in_country_partner":"Specialty Coffee Association of Indonesia",
    "harvest_year":2017,
    "owner_1":"SanJava Coffee",
    "variety":"Sulawesi"
}

url = "http://localhost:9696/predict"

resp = requests.post(url, json=coffee_sample).json()

print(f"The coffee sample is predicted to have a total cup score over 85: {resp['sample_over_85_points_prediction']}.")
