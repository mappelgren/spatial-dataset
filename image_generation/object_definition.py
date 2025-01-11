from __future__ import print_function

import json
import random
from enum import Enum
import numpy as np

from positioning import generate_position_controlled, generate_position
from positioning_util import check_distance_and_margin


def generate_related_attributes(base_attributes, attributes_to_change: "list[str]", relations: "list[Relation]", loaded_properties):
  relation = random.choice(relations)
  print("base attributes")
  print(base_attributes)
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
  print('attributes after')
  print(attributes)

  return attributes


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


def load_properties(filename, args):
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


class Relation(Enum):
  FULL=0
  HIGH=1
  LOW=2
  NO=3
  RANDOM=4

class ExceededMaxTriesError(Exception):
  pass

voc = [
    'above', 'below',
    'over', 'under',
    'next_to', 'away_from',
    'near_to', 'far_from',
    'left_of', 'right_of',
]

dtype = np.float64
voc_rep_org = [
    # above
    np.array([
        [7.00, 7.66, 8.10, 8.61, 8.19, 7.32, 7.66],
        [6.69, 6.56, 7.66, 8.55, 7.13, 7.16, 6.88],
        [5.63, 6.41, 7.09, 8.53, 7.35, 6.74, 5.53],
        [1.94, 2.16, 1.88, 0.00, 1.97, 1.88, 2.00],
        [1.94, 1.78, 1.66, 1.13, 1.63, 2.41, 1.66],
        [1.81, 1.94, 1.42, 1.03, 1.50, 1.84, 1.58],
        [1.44, 1.38, 1.34, 1.19, 1.34, 2.08, 1.44],
    ], dtype=dtype),
    # below
    np.array([
        [1.50, 1.66, 1.29, 1.03, 1.33, 1.75, 1.59],
        [1.71, 2.09, 1.40, 1.31, 1.44, 1.66, 1.45],
        [1.94, 2.09, 1.65, 1.72, 1.88, 2.39, 2.00],
        [2.16, 2.29, 2.03, 0.00, 2.41, 1.94, 2.00],
        [5.66, 6.31, 6.94, 8.16, 6.94, 6.00, 5.81],
        [6.00, 7.09, 7.74, 8.71, 7.78, 7.10, 6.88],
        [7.42, 5.00, 6.88, 8.40, 7.72, 7.71, 7.53],
    ], dtype=dtype),
    # over
    np.array([
        [8.84, 7.65, 8.10, 8.90, 7.59, 7.38, 7.10],
        [6.75, 6.94, 7.19, 8.29, 7.45, 7.32, 8.41],
        [5.69, 5.97, 7.07, 8.42, 7.19, 6.38, 5.58],
        [1.91, 2.19, 2.09, 0.00, 2.13, 1.94, 2.25],
        [2.28, 1.91, 1.71, 1.28, 1.97, 2.09, 2.00],
        [1.69, 2.00, 1.28, 1.45, 2.19, 1.69, 1.66],
        [1.52, 1.59, 1.52, 1.20, 1.28, 1.66, 1.66],
    ], dtype=dtype),
    # under
    np.array([
        [1.81, 1.94, 1.38, 1.39, 1.59, 1.72, 1.47],
        [1.83, 1.53, 2.03, 1.41, 1.44, 1.63, 1.84],
        [1.77, 1.78, 1.63, 1.44, 1.59, 1.68, 2.19],
        [2.06, 2.22, 1.91, 0.00, 2.25, 2.39, 2.00],
        [5.71, 5.66, 6.75, 8.23, 6.84, 5.88, 5.84],
        [6.59, 7.00, 7.59, 7.45, 7.38, 6.50, 6.10],
        [7.22, 7.55, 7.90, 8.72, 7.78, 7.74, 7.03],
    ], dtype=dtype),
    # next_to
    np.array([
        [2.65, 2.06, 2.10, 2.03, 2.29, 1.94, 1.70],
        [2.84, 3.32, 3.31, 3.91, 3.35, 3.34, 2.94],
        [4.06, 4.75, 5.90, 6.70, 6.57, 4.72, 3.87],
        [4.52, 6.00, 8.17, 0.00, 8.39, 6.69, 4.88],
        [3.56, 4.59, 6.59, 6.19, 5.91, 5.38, 4.13],
        [2.94, 3.58, 3.66, 4.06, 4.00, 3.32, 3.06],
        [2.37, 2.06, 2.53, 2.31, 1.81, 2.00, 1.69],
    ], dtype=dtype),
    # away_from
    np.array([
        [7.38, 7.94, 7.45, 7.74, 7.72, 8.10, 8.44],
        [7.41, 6.84, 5.74, 5.16, 5.69, 6.72, 7.22],
        [5.90, 4.75, 2.94, 2.91, 2.78, 5.13, 6.47],
        [5.35, 4.38, 2.13, 0.00, 1.88, 4.58, 6.25],
        [6.32, 4.81, 3.09, 2.50, 3.44, 5.41, 6.45],
        [7.28, 6.09, 5.34, 4.97, 5.41, 5.75, 7.66],
        [8.10, 7.50, 7.58, 7.63, 7.44, 7.83, 8.26],
    ], dtype=dtype),
    # near_to
    np.array([
        [1.74, 1.90, 2.84, 3.16, 2.34, 1.81, 2.13],
        [2.61, 3.84, 4.66, 4.97, 4.90, 3.56, 3.26],
        [4.06, 5.56, 7.55, 7.97, 7.29, 4.80, 3.91],
        [4.47, 5.91, 8.52, 0.00, 7.90, 6.13, 4.63],
        [3.47, 4.81, 6.94, 7.56, 7.31, 5.59, 3.63],
        [3.25, 4.03, 4.50, 4.78, 4.41, 3.47, 3.10],
        [1.84, 2.23, 2.03, 3.06, 2.53, 2.13, 2.00],
    ], dtype=dtype),
    # far_from
    np.array([
        [7.48, 7.94, 7.56, 7.42, 7.38, 7.88, 8.48],
        [6.56, 5.78, 5.41, 5.41, 5.19, 5.38, 7.03],
        [5.69, 4.03, 2.28, 1.78, 2.84, 4.13, 6.06],
        [5.59, 3.44, 1.87, 0.00, 1.66, 4.22, 5.71],
        [6.90, 4.56, 2.28, 1.81, 2.31, 4.09, 6.13],
        [7.09, 6.03, 4.88, 5.19, 5.16, 6.00, 7.42],
        [7.68, 7.77, 7.58, 7.13, 7.47, 7.78, 8.41],
    ], dtype=dtype),
    # left
    np.array([
        [6.56, 5.65, 5.28, 2.56, 2.13, 1.88, 1.66],
        [7.00, 6.06, 5.39, 2.25, 2.16, 1.53, 1.80],
        [7.13, 6.52, 6.34, 2.31, 2.47, 1.94, 2.10],
        [8.35, 7.83, 8.06, 0.00, 1.10, 1.59, 1.94],
        [6.84, 6.39, 6.65, 2.16, 2.03, 2.10, 1.41],
        [6.03, 6.23, 5.63, 2.48, 1.90, 2.22, 1.59],
        [6.16, 5.77, 4.94, 1.90, 1.94, 1.94, 2.03],
    ], dtype=dtype),
    # right
    np.array([
        [1.72, 1.97, 1.66, 2.22, 5.50, 6.45, 6.59],
        [1.90, 2.00, 2.00, 2.28, 5.78, 6.52, 7.06],
        [1.33, 1.63, 2.13, 2.39, 6.39, 6.84, 7.03],
        [1.09, 1.35, 1.38, 0.00, 8.35, 8.52, 8.71],
        [1.69, 1.74, 2.25, 2.09, 6.03, 6.81, 7.50],
        [1.66, 1.94, 1.81, 2.03, 5.59, 6.72, 6.63],
        [1.50, 1.72, 1.94, 1.87, 5.47, 6.13, 6.44],
    ], dtype=dtype),
]

direction_dict = {name: array for name, array in zip(voc, voc_rep_org)}



def generate_object(attributes, scene_struct, positions,
                    fixed_object=None, direction='left',
                    args=None):

  if fixed_object is not None:
    x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
  else:
    x, y, theta = generate_position()
  succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct, args)

  tries = 0

  while not succeeded:
    if fixed_object is not None:
      x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
    else:
      x, y, theta = generate_position()
    succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct, args)
    tries += 1
    if tries > 100:
      raise ExceededMaxTriesError()

  return x, y, theta
