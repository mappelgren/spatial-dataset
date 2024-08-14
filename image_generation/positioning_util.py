from __future__ import print_function

import math

import utils
import bpy, bpy_extras


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


def attributes_from_object(obj):
    return {

    }

def place_object(attributes, x, y, theta, positions, objects, blender_objects, camera, args):
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
    object = {
      'shape': attributes['object'][1],
      'size': attributes['size'][0],
      'material': attributes['material'][1],
      '3d_coords': obj.location,
      'rotation': theta,
      'pixel_coords': {0:pixel_coords},
      'color': attributes['color'][0],
    }
    objects.append(object)
    return object


def check_distance_and_margin(positions, x, y, r, scene_struct, args):
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
