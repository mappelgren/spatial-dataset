#!/bin/bash

cd /srv/image_generation
blender --background --python render_images.py -- --num_images 10 --use_gpu 1
