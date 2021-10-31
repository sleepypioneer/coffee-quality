IMG_NAME := coffee-quality/analysis
DOCKER_BUILD_CONTEXT="."

.PHONY: build
build:
	docker build \
	-f $(DOCKER_BUILD_CONTEXT)/Dockerfile \
	-t $(IMG_NAME) .

.PHONY: dev
dev: build
	docker run -i --network host \
		-v "$(shell pwd):/work" -w /work \
		-e PYTHONPATH=/work/src \
		$(IMG_NAME) \
		jupyter lab \
			--ip='0.0.0.0' --port=8888 \
			--no-browser --allow-root \
			--NotebookApp.token='' --NotebookApp.password=''
