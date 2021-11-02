FROM python:3.9.5-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["./models", "./models"]
COPY ["predict.py", "./"]

ENTRYPOINT ["gunicorn", "predict:app"]