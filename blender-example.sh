#!/bin/bash
cd /srv/image_generation
#blender --background --python render_images.py -- --num_images 2000 --start_idx 0 --use_gpu 1 \
#        --output_image_dir ../one_square/images --output_scene_dir ../one_square/scenes \
#        --height 320 --width 320 --render_function one 

blender --background --python render_images.py -- --num_images 2000 --start_idx 2000 --use_gpu 1 \
        --output_image_dir ../two_mirrored_square/images --output_scene_dir ../two_mirrored_square/scenes \
        --height 320 --width 320 --render_function two_spatial 


#blender --background --python render_images.py -- --num_images 2000 --start_idx 0 --use_gpu 1 \
#        --output_image_dir ../one_square_valid/images --output_scene_dir ../one_square_valid/scenes \
#        --height 320 --width 320 --render_function one 
