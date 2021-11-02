IMG_NAME := coffee-quality-prediction
IMG_TAG := dev
DOCKER_BUILD_CONTEXT="."

.PHONY: build
build:
	docker build \
	-f $(DOCKER_BUILD_CONTEXT)/Dockerfile \
	-t $(IMG_NAME):$(IMG_TAG) .

.PHONY: run-with-docker
run-with-docker: build
	docker run -p 8000:8000 -it $(IMG_NAME):$(IMG_TAG) --bind 0.0.0.0

.PHONY: train
train:
	pipenv run python train.py

.PHONY: dev
dev:
	pipenv run gunicorn --bind 0.0.0.0:8000 predict:app

.PHONY: request
request:
	pipenv run python request.py

.PHONY: create-app
create-app:
	heroku create --region eu coffee-quality-prediction

.PHONY: push-app
push-app:
	heroku container:push web -a coffee-quality-prediction

.PHONY: release-app
release-app:
	heroku container:release web -a coffee-quality-prediction