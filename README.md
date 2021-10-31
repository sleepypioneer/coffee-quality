# Coffee Quality prediction ☕☕☕

This repository was created as part of the [Machine Learning Bootcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev). This project has been submitted as the midterm project for the course. It take a data set about coffee quality with various attributes around the raw coffee including it's origin and processing as well as the assigned quality scores. I chose this dataset as before working as a software engineer I was a coffee roaster so I have pre existing domain knowledge and a passion for coffee ☕ If you notice any mistakes/ improvements to the code feel free to open an issue 💖

## Predicting Coffee quality - the problem we are trying to solve 🕵️‍♀️

Coffee quality is assessed by tastings called [cuppings](https://www.baristainstitute.com/inspiration/what-coffee-cupping) where the coffee is scored on various qualities such as aroma, balance, sweetness and cleanliness of the flavour. These scores create a total cupping score. Speciality coffees are normally those which score over 85 points and coffee over 90 are considered exceptionally high quality. The quality of a coffee is dependant on a number of factors including the varietal, growing conditions, processing method as well as the storage of the green sample before roasting.

Predicting quality before roasting could be useful for farmers who often don't have the means to cup their own coffee or for buyers looking to narrow down the selection of coffee they are tasting based on a range of scores (ie only tasting coffees predicted to be over 90 points or between 85 - 90)

This project focuses on predicting the total score, however it could also be interesting to look at the individual scores to predict for example a coffee's sweetness or aroma score and potentially offer recommendations on coffees based on a user's preference.


## Navigating the project repository 🗂️

Where to find the files for evaluation :)

- 📂 **Analysis**
    - [this directory](./analysis/README.md) has all the notebooks for exploring the data as well as building, tuning and evaluating the model.
    - [Data preparation and EDA notebook](./analysis/notebooks/data_exploration.ipynb)
    - [Model selection notebook](./analysis/notebooks/model_training.ipynb)
- 📂 **App**
    - [this directory](./app/README.md) contains the code for the flask websever which can be used to make predictions on new samples using the final trained model. Here you will find the scripts [train.py](./app/train.py) and [predict.py](./app/predict.py) as well as the [Pipenv](./app/Pipfile) and [Docker](./app/Dockerfile) files for running the service.
- 📂 **Deployment**
    - this project has been deployed [here](https://coffee-quality-prediction.herokuapp.com/predict) on Heroku using the instructions kindly shared by [Ninad](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md) in the course slack channel. This end point will remain available until the end of the evaluation period.
    - example request to this end point would be:

    ```sh
        curl --header "Content-Type: application/json" \
            --request POST \
            --data '{"username":"xyz","password":"xyz"}' \
            https://coffee-quality-prediction.herokuapp.com/predict
    ```


## Running the project ▶️

### Requirements ⚙️

I advise using a virtual environment for running this project, below are instructions for doing so using [venv](https://docs.python.org/3/library/venv.html) which you can install on linux with the following command `pip install venv`. Additionally if you would like to run the analysis notebooks or the app in Docker you will need to have [Docker](https://docs.docker.com/get-docker/) installed.

### Start a virtual environment 🌐

```sh
# create virtual environment
python3 -m venv coffee-quality

# start the virtual environment
source coffee-quality/bin/activate

# install pipenv required for the app's dependency management
pip install pipenv
```

Then go to the respective README for further instructions on running either the [analysis notebooks](./analysis/README.md) or the [prediction flask application](./app/README.md)

## Data 💽

The data used for this project is gathered from [Coffee Quality Institute (CQI)](https://database.coffeeinstitute.org/) in January, 2018. Scraping was performed by [James LeDoux](https://github.com/jldbc) and more details can be found [here](https://github.com/jldbc/coffee-quality-database)

## Linting your code ✔️
We are linting the project with Black and Flake8 we reccomend running these both locally before pushing code as they are enforced in the github actions.

## Github actions 🎬

When pushing code to github it will run actions for linting the code, you can find, add and update these actions in /.github/workflows.