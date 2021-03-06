# Coffee Quality prediction app ☕

This code runs a small flask based webserver which can make predictions on the given coffee samples total cupping score provided the following attributes are passed as a JSON.

## Training the model

The script [train.py](./train.py) can be used to train a model with the final model and parameters. After running it will print the current model and parameters as well as the auc and rsme scores for the model. A `model.bin` artifact will be saved in the `./models` directory. Note that these are ignored by git and therefore not saved to github. **You need to run this before you can run the application**.

## Running locally 🖥️

### Install dependencies ⚙️

First set up the virtual environment as documented in the root [README](../README.md). Then run:

```sh
pipenv install
```

### Running the application ▶️

The below command will run the application locally within the virtual environment. 

```sh
make dev
```

## Running with Docker 🐳

The below command will both build the Docker image and run it.

```sh
make run-with-docker
```

## Example call to the service

### Using Curl

```sh
curl --header "Content-Type: application/json" \
    --request POST \
    http://localhost:8000/predict \
    -d @test_data/coffee_sample.json
```

### Using Python

```python
import requests

coffee_sample = {
    "moisture": "0.12",
    ...
}

url = "http://localhost:8000/predict"

requests.post(url, json=coffee_sample).json()
# returns python dictionary of response
```

There is a sample code for this [here](./request.py) which you can run with `pipenv run python request.py`.

## Deployment with Heroku

Run the following commands to deploy this project to [Heroku](https://dashboard.heroku.com/login).

```sh
# login into Heroku
heroku login
heroku container:login

# only need to run this the first time to create the app
make create-app

make push-app

make release-app
``