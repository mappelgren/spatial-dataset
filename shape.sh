#!/bin/bash
cd /srv/image_generation
blender --background --python render_images.py -- --num_images 1 --start_idx 0 --use_gpu 1 \
	--output_image_dir ../shape/images --output_scene_dir ../shape/scenes \
        --height 320 --width 320 --render_function one 

#blender --background --python render_images.py -- --num_images 2000 --start_idx 2000 --use_gpu 1 \
#        --output_image_dir ../shape/images --output_scene_dir ../shape/scenes \
#        --height 320 --width 320 --render_function one 


#blender --background --python render_images.py -- --num_images 2000 --start_idx 0 --use_gpu 1 \
#        --output_image_dir ../shape_valid/images --output_scene_dir ../shape_valid/scenes \
#        --height 320 --width 320 --render_function one 
