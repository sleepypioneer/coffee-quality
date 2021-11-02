# Coffee Quality prediction app ☕

This code runs a small flask based webserver which can make predictions on the given coffee samples total cupping score provided the following attributes are passed as a JSON.

## Training the model

The script [train.py]() can be used to train a model with the final model and parameters. After running it will print the current model and parameters as well as the auc and rsme scores for the model. A `model.bin` artifact will be saved in the `./models` directory. Note that these are ignored by git and therefore not saved to github. **You need to run this before you can run the application**.

## Running locally 🖥️

### Install dependencies ⚙️

First set up the virtual environment as documented in the root [README](../README.md). Then run:

```sh
pipenv install
```

### Running the application ▶️

```sh
make dev
```

## Running with Docker 🐳

```sh
make run-with-docker
```

## Example call to the service

### Using Curl

```sh
curl --header "Content-Type: application/json" \
    --request POST \
    http://localhost:9696/predict \
    --data '
        {"moisture":0.12,
        "quakers":0.0,
        "category_two_defects:15,
        "species":"Arabica",
        "owner":"sanjava coffee",
        "farm_name":"various",
        "company":"pt. shriya artha nusantara",
        "region":"sapan toraja",
        "producer":"vary farm",
        "in_country_partner":"Specialty Coffee Association of Indonesia",
        "harvest_year":2017,
        "owner_1":"SanJava Coffee",
        "variety":"Sulawesi"}
    '
    EOT
```

data = {"moisture":0.12,
"quakers":0.0,
"category_two_defects:15,
"species":"Arabica",
"owner":"sanjava coffee",
"farm_name":"various",
"company":"pt. shriya artha nusantara",
"region":"sapan toraja",
"producer":"vary farm",
"in_country_partner":"Specialty Coffee Association of Indonesia",
"harvest_year":2017,
"owner_1":"SanJava Coffee",
"variety":"Sulawesi"}

### Using Python

```python
import requests

coffee_sample = {
    "gender": "female",
    ...
}

url = "http://localhost:9696/predict"

requests.post(url, json=coffee_sample).json()
# returns python dictionary of response
```

There is a sample code for this [here](./request.py) which you can run with `pipenv run python request.py`.
