from __future__ import print_function

import json
import random
from enum import Enum

from positioning import generate_position_controlled, generate_position
from positioning_util import check_distance_and_margin


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


def generate_object(loaded_properties, scene_struct, positions,
                    fixed_object=None, fix_target=False, direction='left', args=None):

  attributes = generate_random_attributes(loaded_properties, fix_target=fix_target)
  if fixed_object is not None:
    x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
  else:
    x, y, theta = generate_position()
  succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct, args)

  while not succeeded:
    if fixed_object is not None:
      x, y, theta = generate_position_controlled(scene_struct, fixed_object, direction=direction)
    else:
      x, y, theta = generate_position()
    succeeded = check_distance_and_margin(positions, x, y, attributes['size'][1], scene_struct, args)

  return attributes, x, y, theta
