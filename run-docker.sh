#!/bin/bash

docker run --rm -it\
    --runtime=nvidia \
    --gpus 1,2,3 \
    --name=blender-2.79b_cuda-8 \
    -v/home/xappma/spatial-dataset:/srv \
    blender-2.79b-cuda-8:latest \
    /srv/color.sh
