 # Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function

import copy
from enum import Enum
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import numpy as np
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

class Relation(Enum):
  FULL=0
  HIGH=1
  LOW=2
  NO=3
  RANDOM=4

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

def rotateMatrix(a):
   return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def generate_point(direction_vector, origin):
  x, y = direction_vector[:2]
  angle = np.arctan2(y, x)
  print(angle)
  rot = rotateMatrix(angle)
  p = [-6, -6]
  while not in_bounds(p):
    x = random.uniform(-6, 6)
    y = random.uniform(0, 6)
    p = [x, y]
    p = p @ rot + [origin[0], origin[1]]
  return p

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

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
      'groups': {}
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)


  # Add random jitter to camera position
  # if args.camera_jitter > 0:
  #   for i in range(3):
  #     bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)



  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']

  print(camera)
  print(camera.matrix_world)
  print(camera.matrix_world.to_quaternion())
  print(camera.location)
  print(camera.rotation_euler)



  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)




  # Now make some random objects
  objects, blender_objects, object_indices = add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  scene_struct['groups'] = object_indices
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  originalx, originaly = camera.location.x, camera.location.y
  for degree, file_name in zip([90, 180, 270], output_rot_images):

    xy = np.array([originalx, originaly]) @ rotateMatrix(np.radians(degree))
    x, y = xy
    camera.location.x = x
    camera.location.y = y


    render_args.filepath = file_name

    if degree == 90:
      ground = bpy.data.objects['Ground']

      ground.location.x = -15
      ground.location.y = 25

      ground.rotation_euler[2] = math.radians(64)

    elif degree == 180:
      ground = bpy.data.objects['Ground']

      ground.location.x = 24
      ground.location.y = 22

      ground.rotation_euler[2] = math.radians(334)

    elif degree == 270:
      ground = bpy.data.objects['Ground']

      ground.location.x = 15
      ground.location.y = -24

      ground.rotation_euler[2] = math.radians(244)



    while True:
      try:
        bpy.ops.render.render(write_still=True)
        break
      except Exception as e:
        print(e)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def load_properties(filename):
  # Load the property file
  loaded_properties = {}
  with open(filename, 'r') as f:
    properties = json.load(f)
    loaded_properties['color_name_to_rgba'] = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      loaded_properties['color_name_to_rgba'][name] = rgba
    loaded_properties['material_mapping'] = [(v, k) for k, v in properties['materials'].items()]
    loaded_properties['object_mapping'] = [(v, k) for k, v in properties['shapes'].items()]
    loaded_properties['size_mapping'] = list(properties['sizes'].items())

  loaded_properties['shape_color_combos'] = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      loaded_properties['shape_color_combos'] = list(json.load(f).items())

  return loaded_properties


def generate_object(loaded_properties, scene_struct, positions,
                    fixed_object=None, fix_target=False, direction='left'):

  attributes = generate_random_attributes(loaded_properties, fix_target=fix_target)
  if fixed_object is not None:
    x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
  else:
    x, y, theta = generate_position()
  succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct)
  while not succeeded:
    if fixed_object is not None:
      x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
    else:
      x, y, theta = generate_position()
    succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct)



  return attributes, x, y, theta

def add_objects(scene_struct, num_target_group, num_non_target_group, args, camera):
  """
  Add target objects, target group objects and non-target group objects to the current blender scene
  """

  loaded_properties = load_properties(args.properties_json)

  positions = []
  objects = []
  blender_objects = []
  group_indices = {
    'target': [],
    'target_group': [],
    'non_target_group': []
  }

  distractor = generate_random_attributes(loaded_properties)
  x, y, theta = generate_position()
  place_object(distractor, x, y, theta, positions, objects, blender_objects, camera)

  p1 = [x, y]

  distractor = objects[0]

  target_attributes, x, y, theta = generate_object(loaded_properties, scene_struct, positions,
                                                   fix_target=True, fixed_object=distractor, direction='left')
  p2 = [x, y]
  place_object(target_attributes, x, y, theta, positions, objects, blender_objects, camera)

  target_attributes, x, y, theta = generate_object(loaded_properties, scene_struct, positions,
                                                   fix_target=True, fixed_object=distractor, direction='right')
  p3 = [x, y]
  place_object(target_attributes, x, y, theta, positions, objects, blender_objects, camera)

  # plot(p1, p2, p3, scene_struct['directions']['left'], scene_struct['directions']['right'])

  print('points')
  print(p1)
  print(p2)
  print(p3)


  #
  # # target object
  # target_attributes = generate_random_attributes(loaded_properties, fix_target=True)
  # # x, y, theta = generate_position()
  # x, y, theta = generate_position()
  # place_object(target_attributes, x, y, theta, positions, objects, blender_objects, camera)
  #



  # # keeps track, which object is added
  # object_index = 0
  # group_indices['target'].append(object_index)
  #
  #
  # # target group
  # for i in range(1, num_target_group + 1):
  #   object_index += 1
  #   object_attributes = generate_related_attributes(target_attributes,
  #                                                   args.target_group_attributes,
  #                                                   args.target_group_relations,
  #                                                   loaded_properties)
  #
  #   num_tries = 0
  #   while True:
  #     num_tries += 1
  #     if num_tries > args.max_retries:
  #       for obj in blender_objects:
  #         utils.delete_object(obj)
  #       return add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)
  #
  #     x, y, theta = generate_position_controlled(scene_struct, objects[0])
  #     succeded = check_distance_and_margin(positions, x, y, object_attributes['size'][1], scene_struct)
  #
  #     if succeded:
  #       break
  #
  #   place_object(object_attributes, x, y, theta, positions, objects, blender_objects, camera)
  #   group_indices['target_group'].append(object_index)
  #
  # # non-target group
  # for i in range(1, num_non_target_group + 1):
  #   object_index += 1
  #   object_attributes = generate_related_attributes(target_attributes,
  #                                                   args.non_target_group_attributes,
  #                                                   args.non_target_group_relations,
  #                                                   loaded_properties)
  #
  #   num_tries = 0
  #   while True:
  #     num_tries += 1
  #     if num_tries > args.max_retries:
  #       for obj in blender_objects:
  #         utils.delete_object(obj)
  #       return add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)
  #
  #     x, y, theta = generate_position_controlled(scene_struct, objects[0])
  #     succeded = check_distance_and_margin(positions, x, y, object_attributes['size'][1], scene_struct)
  #
  #     if succeded:
  #       break
  #
  #   place_object(object_attributes, x, y, theta, positions, objects, blender_objects, camera)
  #   group_indices['non_target_group'].append(object_index)


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_objects(scene_struct, num_target_group, num_non_target_group, args, camera)

  return objects, blender_objects, group_indices

def get_coords(vector, origin, length, negative=True):
  if negative:
    x = np.linspace(-length, length)
  else:
    x = np.linspace(0, length)
  xs = [x0 * vector[0] + origin[0] for x0 in x]
  ys = [y0 * vector[1] + origin[1] for y0 in x]
  return xs, ys


def generate_position():
  #TODO This seems like the place to decide where the object is placed
  x = random.uniform(-3, 3)
  y = random.uniform(-3, 3)
  # Choose random orientation for the object.
  theta = 360.0 * random.random()

  return x, y, theta


def in_bounds(point):
  x, y = point

  return -3 <= x <= 3 and -3 <= y <= 3


def line_from_coord(dir_vector, origin):
  x1, y1, _ = origin
  dx, dy, _ = dir_vector
  x2, y2 = x1 + dx, y1 + dy
  p1 = (x1, y1)
  p2 = (x2, y2)

  A = (p1[1] - p2[1])
  B = (p2[0] - p1[0])
  C = (p1[0] * p2[1] - p2[0] * p1[1])

  return A, B, -C

def get_coords(vector, origin, length):
  x = np.linspace(-length, length)
  xs = [x0 * vector[0] + origin[0] for x0 in x]
  ys = [y0 * vector[1] + origin[1] for y0 in x]
  return xs, ys


def is_left(a, b, c):
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0;

def is_right(a, b, c):
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0;


def is_valid(int1, int2, point, direction='left'):
  if direction == 'left':
    return is_left(int1, int2, point) and in_bounds(point)
  elif direction == 'right':
    return is_right(int1, int2, point) and in_bounds(point)
  else:
    raise NotImplementedError()

def intersection(L1, L2):
  D = L1[0] * L2[1] - L1[1] * L2[0]
  Dx = L1[2] * L2[1] - L1[1] * L2[2]
  Dy = L1[0] * L2[2] - L1[2] * L2[0]
  if D != 0:
    x = Dx / D
    y = Dy / D
    return x, y
  else:
    return False

up_absolute = (0, 1, 0)
right_absolute = (1, 0, 0)
boundary_lines = [line_from_coord(up_absolute, (-3, 0, 0)),
                  line_from_coord(up_absolute, (3, 0, 0)),
                  line_from_coord(right_absolute, (0, -3, 0)),
                  line_from_coord(right_absolute, (0, 3, 0))]

def generate_position_controlled(scene_struct, fixed_obj, direction='left'):
  dir_vector = scene_struct['directions'][direction]
  coord = fixed_obj['3d_coords']

  x, y = generate_point(dir_vector, coord)

  # Choose random orientation for the object.
  theta = 360.0 * random.random()
  print(direction, dir_vector)
  print("coord", x, y, theta)


  return x, y, theta

def check_distance_and_margin(positions, x, y, r, scene_struct):
  # Check to make sure the new object is further than min_dist from all
  # other objects, and further than margin along the four cardinal directions
  dists_good = True
  margins_good = True
  for (xx, yy, rr) in positions:
    dx, dy = x - xx, y - yy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist - r - rr < args.min_dist:
      dists_good = False
      break
    for direction_name in ['left', 'right', 'front', 'behind']:
      direction_vec = scene_struct['directions'][direction_name]
      assert direction_vec[2] == 0
      margin = dx * direction_vec[0] + dy * direction_vec[1]
      if 0 < margin < args.margin:
        print(margin, args.margin, direction_name)
        print('BROKEN MARGIN!')
        margins_good = False
        break
    if not margins_good:
      break

  return dists_good and margins_good

def generate_random_attributes(loaded_properties, fix_target=False):
  attributes = {}
  # Choose a random size
  attributes['size'] = random.choice(loaded_properties['size_mapping'])

  if fix_target:
    attributes['size'] = loaded_properties['size_mapping'][0]
    attributes['color'] = 'red', loaded_properties['color_name_to_rgba']['red']
    attributes['material'] = loaded_properties['material_mapping'][0]
    attributes['object'] = loaded_properties['object_mapping'][0]

    return attributes

  # Choose random color and shape
  if loaded_properties['shape_color_combos'] is None:
    attributes['object'] = random.choice(loaded_properties['object_mapping'])
    attributes['color'] = random.choice(list(loaded_properties['color_name_to_rgba'].items()))
  else:
    obj_name_out, color_choices = random.choice(loaded_properties['shape_color_combos'])
    color_name = random.choice(color_choices)
    obj_name = [k for k, v in loaded_properties['object_mapping'] if v == obj_name_out][0]
    rgba = loaded_properties['color_name_to_rgba'][color_name]

    attributes['object'] = obj_name, obj_name_out
    attributes['color'] = color_name, rgba

  attributes['material'] = random.choice(loaded_properties['material_mapping'])

  return attributes

def generate_related_attributes(base_attributes, attributes_to_change: "list[str]", relations: "list[Relation]", loaded_properties):
  relation = random.choice(relations)

  if relation == Relation.RANDOM:
    return generate_random_attributes(loaded_properties)
  
  attributes = base_attributes.copy()
  attributes_to_change = random.sample(attributes_to_change, min(len(attributes_to_change), relation.value))

  for attribute in attributes_to_change:
    if attribute == 'size':
      while attributes['size'] == base_attributes['size']:
        attributes['size'] = random.choice(loaded_properties['size_mapping'])
    elif attribute == 'shape':
      while attributes['object'] == base_attributes['object']:
        attributes['object'] = random.choice(loaded_properties['object_mapping'])
    elif attribute == 'color':
      while attributes['color'] == base_attributes['color']:
        attributes['color'] = random.choice(list(loaded_properties['color_name_to_rgba'].items()))
  # TODO shape_color_combos
  
  return attributes


def place_object(attributes, x, y, theta, positions, objects, blender_objects, camera):
    # For cube, adjust the size a bit
    r = attributes['size'][1]
    if attributes['object'][0] == 'Cube':
      r /= math.sqrt(2)

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, attributes['object'][0], r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    utils.add_material(attributes['material'][0], Color=attributes['color'][1])

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': attributes['object'][1],
      'size': attributes['size'][0],
      'material': attributes['material'][1],
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': attributes['color'][0],
    })

def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


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

