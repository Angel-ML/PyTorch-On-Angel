#!/bin/bash
set -e

export IMAGE_NAME=pytorch_on_angel:artifacts

docker build --target ARTIFACTS -t ${IMAGE_NAME} .
docker run -it --rm -v $(pwd)/dist:/output ${IMAGE_NAME}
echo "***** output files in ./dist *****"
