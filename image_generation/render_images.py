# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function

import sys, random, argparse, json, os
from datetime import datetime as dt

from render_utils import render
from scene_util import load_everything, set_up_camera, set_up_lights, save_scene_struct, rotate_camera

from object_definition import generate_object, ExceededMaxTriesError
from object_definition import generate_random_attributes, load_properties, Relation
from positioning import generate_position
from positioning_util import compute_all_relationships, place_object
from render_utils import check_visibility

# import matplotlib.pyplot as plt

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")
parser.add_argument('--min_number_target_group_objects', default=1, type=int,
    help="The minimum number of objects in the target group.")
parser.add_argument('--max_number_target_group_objects', default=1, type=int,
    help="The maximum number of objects in the target group.")
parser.add_argument('--min_number_non_target_group_objects', default=0, type=int,
    help="The minimum number of objects that are not in the target group.")
parser.add_argument('--max_number_non_target_group_objects', default=0, type=int,
    help="The maximum number of objects that are not in the target group.")
parser.add_argument('--target_group_relations', nargs='*', default=['FULL'],
    help="A list of possible relations between target objects and objects in the target group.")
parser.add_argument('--non_target_group_relations', nargs='*', default=['RANDOM'],
    help="A list of possible relations between target objects and objects not in the target group.")
parser.add_argument('--target_group_attributes', nargs='*', default=['color', 'shape', 'size'],
    help="A list of attributes that can be changed for the tharget group. (The number of attributes that are changed is controlled by --target_group_relations)")
parser.add_argument('--non_target_group_attributes', nargs='*', default=['color', 'shape', 'size'],
    help="A list of attributes that can be changed for the non tharget group. (The number of attributes that are changed is controlled by --non_target_group_relations)")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  imgrot90_template = '%s%%0%dd_rot90.png' % (prefix, num_digits)
  imgrot180_template = '%s%%0%dd_rot180.png' % (prefix, num_digits)
  imgrot270_template = '%s%%0%dd_rot270.png' % (prefix, num_digits)

  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  imgrot90_template = os.path.join(args.output_image_dir, imgrot90_template)
  imgrot180_template = os.path.join(args.output_image_dir, imgrot180_template)
  imgrot270_template = os.path.join(args.output_image_dir, imgrot270_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  all_scene_paths = []
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    img90_path = imgrot90_template % (i + args.start_idx)
    img180_path = imgrot180_template % (i + args.start_idx)
    img270_path = imgrot270_template % (i + args.start_idx)

    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    
    num_target_group = random.randint(args.min_number_target_group_objects, args.max_number_target_group_objects)
    num_non_target_group = random.randint(args.min_number_non_target_group_objects, args.max_number_non_target_group_objects)

    render_scene(args,
      num_target_group=num_target_group,
      num_non_target_group=num_non_target_group,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      output_rot_images=[img90_path, img180_path, img270_path])

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

def render_scene(args,
    num_target_group=2,
    num_non_target_group=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    output_rot_images=['render_rot.png']
  ):

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
    'split': output_split,
    'image_index': output_index,
    'image_filename': os.path.basename(output_image),
    'objects': [],
    'directions': {},
    'groups': {}
  }

  #Set up renderer and camera
  render_args = load_everything(args, output_image)
  camera = set_up_camera(args, scene_struct)
  set_up_lights(args)

  # Now make some random objects
  add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)

  render(output_image, render_args)

  original_x, original_y = camera.location.x, camera.location.y

  for degree, file_name in zip([90, 180, 270], output_rot_images):

    rotate_camera(camera, original_x, original_y, degree)

    render(file_name, render_args)

    for obj in scene_struct['objects']:
        pixel_coords = utils.get_camera_coords(camera, obj['3d_coords'])
        obj['pixel_coords'][degree] = pixel_coords

  save_scene_struct(output_scene, scene_struct)

  # if output_blendfile is not None:
  #   bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_object_random_position(args, fixed_attributes=None):
    if fixed_attributes is None:
        attributes = generate_random_attributes(args['properties'])
    else:
        attributes = fixed_attributes
    x, y, theta = generate_object(attributes, args['scene_struct'], args['positions'], args=args['args'])
    return place_object(attributes, x, y, theta, args['positions'], args['objects'],
                        args['blender_objects'], args['camera'], args['args']), attributes

def add_object(fixed_object, direction, args, fixed_attributes=None):
    if fixed_attributes is None:
        attributes = generate_random_attributes(args['properties'])
    else:
        attributes = fixed_attributes
    x, y, theta = generate_object(attributes, args['scene_struct'], args['positions'],
                                  fixed_object=fixed_object, direction=direction, args=args['args'])
    return place_object(attributes, x, y, theta, args['positions'], args['objects'],
                        args['blender_objects'], args['camera'], args['args']), attributes

directions = ['left', 'right', 'front', 'behind']
opposite = {'left':'right', 'right':'left', 'front':'behind', 'behind':'front'}


def one_in_middle(args):
    dir = random.choice(directions)

    distractor, _ = add_object_random_position(args)
    _, target_attributes = add_object(distractor, dir, args)
    add_object(distractor,opposite[dir], args, fixed_attributes=target_attributes)


def one_splayed(args):
    dir1 = dir2 = random.choice(directions)
    while dir2 == dir1:
        dir2 = random.choice(directions)

    distractor, _ = add_object_random_position(args)
    _, target_attributes = add_object(distractor, dir1, args)
    add_object(distractor, dir2, args, fixed_attributes=target_attributes)


def two_ambiguous(args):

    o1, attributes = add_object_random_position(args)
    add_object_random_position(args, fixed_attributes=attributes)

def row(args):
    dir = random.choice(directions)

    distractor, distractor_attributes = add_object_random_position(args)
    o1, target_attributes = add_object(distractor, dir, args)
    o2, _ = add_object(o1, dir, args, fixed_attributes=distractor_attributes)
    add_object(o2, dir, args, fixed_attributes=target_attributes)



def add_objects(scene_struct, num_target_group, num_non_target_group, args, camera):
  """
  Add target objects, target group objects and non-target group objects to the current blender scene
  """

  loaded_properties = load_properties(args.properties_json, args)

  positions = []
  objects = []
  blender_objects = []
  group_indices = {
    'target': [],
    'target_group': [],
    'non_target_group': []
  }

  adding_object_args = {'properties':loaded_properties, 'scene_struct':scene_struct, 'positions': positions,
                        'objects':objects, 'blender_objects':blender_objects, 'camera':camera, 'args':args}

  try:
     one_in_middle(adding_object_args)
    # two_ambiguous(adding_object_args)
    # row(adding_object_args)
  except ExceededMaxTriesError:
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)

  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  scene_struct['groups'] = group_indices

  # return objects, blender_objects, group_indices


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    args.target_group_relations = [Relation[arg] for arg in args.target_group_relations]
    args.non_target_group_relations = [Relation[arg] for arg in args.non_target_group_relations]
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

