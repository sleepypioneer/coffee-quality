# Coffee Quality prediction ☕☕☕

This repository was created as part of the [Machine Learning Bootcamp]() by [Alexey Grigov](). This project has been submitted as the midterm project for the course. It take a data set about coffee quality with various attributes around the raw coffee including it's origin and processing as well as the assigned quality scores. I chose this dataset as before working as an engineer I was a coffee roaster so I have pre existing domain knowledge and a passion for coffee ☕

## Navigating the project repository

- **Analysis**
    - [this directory](./analysis/README.md) has all the notebooks for exploring the data as well as building, tuning and evaluating the model.
- **App**
    - [this directory](./app/README.md) contains the contain for the flask websever which can be used to make predictions on new samples using the exported model
- **Deployment**
    - this project has been deployed [here]()


## Running the project

Start a virtual environment

```sh
# create virtual environment
python3 -m venv coffee-quality

# start the virtual environment
source coffee-quality/bin/activate

# install pipenv required for the app's dependency management
pip install pipenv
```