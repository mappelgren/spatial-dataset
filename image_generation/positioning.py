from __future__ import print_function

import random
import numpy as np

def generate_position_controlled(scene_struct, fixed_obj, direction='left'):
  dir_vector = scene_struct['directions'][direction]
  coord = fixed_obj['3d_coords']

  x, y = generate_point(dir_vector, coord)

  # Choose random orientation for the object.
  theta = 360.0 * random.random()
  print(direction, dir_vector)
  print("coord", x, y, theta)


  return x, y, theta


# up_absolute = (0, 1, 0)
# right_absolute = (1, 0, 0)
# boundary_lines = [line_from_coord(up_absolute, (-3, 0, 0)),
#                   line_from_coord(up_absolute, (3, 0, 0)),
#                   line_from_coord(right_absolute, (0, -3, 0)),
#                   line_from_coord(right_absolute, (0, 3, 0))]
def in_bounds(point):
  x, y = point

  return -3 <= x <= 3 and -3 <= y <= 3


def generate_position():
  #TODO This seems like the place to decide where the object is placed
  x = random.uniform(-3, 3)
  y = random.uniform(-3, 3)
  # Choose random orientation for the object.
  theta = 360.0 * random.random()

  return x, y, theta


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


def rotateMatrix(a):
   return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
