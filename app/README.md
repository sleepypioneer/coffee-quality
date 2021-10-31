# Coffee Quality prediction app ☕

This code runs a small flask based webserver which can make predictions on the given coffee samples total cupping score provided the following attributes are passed as a JSON.

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
    --data '{"username":"xyz","password":"xyz"}' \
    http://localhost:9696/predict

```

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