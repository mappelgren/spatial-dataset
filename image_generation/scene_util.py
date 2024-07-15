from __future__ import print_function

import json
import math
import random

import render_utils
import utils
from positioning import rotateMatrix

import bpy, bpy_extras
from mathutils import Vector
import numpy as np

def load_everything(args, output_image):
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

  return render_args


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def set_up_camera(args, scene_struct):
  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object



  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']

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

  return camera


def set_up_lights(args):
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


def save_scene_struct(output_scene, scene_struct):
  with open(output_scene, 'w') as f:
    for obj in scene_struct['objects']:
      obj['3d_coords'] = tuple(obj['3d_coords'])
    json.dump(scene_struct, f, indent=2)


def rotate_camera(camera, originalx, originaly, degree):
    xy = np.array([originalx, originaly]) @ rotateMatrix(np.radians(degree))
    x, y = xy
    camera.location.x = x
    camera.location.y = y

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