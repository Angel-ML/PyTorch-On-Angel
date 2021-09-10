#!/bin/bash
set -e

IMAGE_NAME=pytorch_on_angel:dev
MODEL_PATH=$1

docker build --target DEV -t ${IMAGE_NAME} .
echo "***** generating pytorch script model for ${MODEL_PATH}, output in ./dist *****"
docker run -it --rm -v $(pwd)/python/graph/nn:/nn -v $(pwd)/python/graph/utils:/utils -v $(pwd)/${MODEL_PATH}:/model.py -v $(pwd)/dist:/output -w /output ${IMAGE_NAME} python /model.py ${@:2}
