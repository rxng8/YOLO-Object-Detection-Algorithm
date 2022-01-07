import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import cv2

from .const import *

def iou(A, B):
  # xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2

  xmin1, ymin1, xmax1, ymax1 = A
  xmin2, ymin2, xmax2, ymax2 = B

  intersection = (min(xmax1, xmax2) - max(xmin1, xmin2)) * (min(ymax1, ymax2) - max(ymin1, ymin2))
  intersection = max(0, intersection)
  union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
  return float(intersection + 1e-9) / (union + 1e-9)

def dynamic_iou(A, B):
  """

  Args:
      A ([type]): tensor (e.g) with shape (batch_size, n_cell_y, n_cell_x, 4)
        the last dimension is expected to be center_x, center_y, width, heigth
      B ([type]): tensor (e.g) with shape (batch_size, n_cell_y, n_cell_x, 4)

  Return:
    (Tensor): with shape (batch_size, n_cell_y, n_cell_x)
  """
  # (batch_size, n_cell_y, n_cell_x, 2)
  A_wh = A[..., 2:4]
  A_wh_half = A_wh / 2.
  A_box_xy = A[..., 0:2]
  A_mins = A_box_xy - A_wh_half
  A_maxs = A_box_xy + A_wh_half

  B_wh = B[..., 2:4]
  B_wh_half = B_wh / 2.
  B_box_xy = B[..., 0:2]
  B_mins = B_box_xy - B_wh_half
  B_maxs = B_box_xy + B_wh_half

  intersect_mins = tf.maximum(A_mins, B_mins) # (batch_size, n_cell_y, n_cell_x, 2)
  intersect_maxs = tf.minimum(A_maxs, B_maxs) # (batch_size, n_cell_y, n_cell_x, 2)
  intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0) # (batch_size, n_cell_y, n_cell_x, 2)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # (batch_size, n_cell_y, n_cell_x)

  A_areas = A_wh[..., 0] * A_wh[..., 1] # (batch_size, n_cell_y, n_cell_x)
  B_areas = B_wh[..., 0] * B_wh[..., 1] # (batch_size, n_cell_y, n_cell_x)

  union_areas = A_areas + B_areas - intersect_areas

  ious = tf.truediv(intersect_areas + EPSILON, union_areas + EPSILON)

  return ious # (batch_size, n_cell_y, n_cell_x)



def test_dynamic_iou():
  pass

def mAP(A, B):
  pass

def test_iou():
  A = [0.35, 0.45, 0.67, 0.87]
  B = [0.38, 0.24, 0.56, 0.98]
  test_value = iou(A, B)
  assert test_value >= 0, test_value <= 1

def show_img(img):
  plt.axis("off")
  plt.imshow(img)
  plt.show()

def preprocess_image(img: np.array, image_size=(320, 320)):
  """ 

  Args:
      img ([type]): Expect image with shape (height, width, 3). With range [0, 255], int
  """
  # Preprocessing image
  preprocessed_img = tf.cast(img, tf.float32)
  preprocessed_img /= 255.0
  assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
  preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
  return preprocessed_img


def draw_boxes(img, boxes: List[List[float]]):
  """

  Args:
      img ([type]): tensorflow Tensor object
      box (List[List[float]]): Expect list of boxes, each box is [x, y, width, heigh]. (x, y) is the top-left point
  """
  img_height, img_width = img.shape[0:2]
  try:
    boxed_img = img.numpy()
  except:
    pass
  for box in boxes:
    x, y, width, height = box
    xmin = int(x * img_width)
    ymin = int(y * img_height)
    xmax = int((x+width) * img_width)
    ymax = int((y+height) * img_height)

    boxed_img = cv2.rectangle(
      boxed_img,
      (xmin, ymin), # (x1, y1)
      (xmax, ymax), # (x2, y2)
      color=(0,1,0),
      thickness=2
    )

  return boxed_img

# Test code
# Print info
# print(f"resized_xmin: {resized_xmin}, \
# resized_ymin: {resized_ymin}, \
# resized_xmax: {resized_xmax}, \
# resized_ymax: {resized_ymax}, \
# width: {width}, \
# height: {height}, \
# center_x: {center_x}, \
# center_y: {center_y}")

# Show the img with bounding boxes for the resized img
# boxed_img = cv2.rectangle(
#   preprocessed_img.numpy(),
#   (int(resized_xmin * image_size[1]), int(resized_ymin * image_size[0])), # (x1, y1)
#   (int(resized_xmax * image_size[1]), int(resized_ymax * image_size[0])), # (x2, y2)
#   color=(0,255,0),
#   thickness=2
# )
# show_img(boxed_img)
# print(boxed_img.shape)

# Show the img with bounding boxes for the original img
# cv2.rectangle(
#   original_img,
#   (int(box["xmin"]), int(box["ymin"])), # (x1, y1)
#   (int(box["xmax"]), int(box["ymax"])), # (x2, y2)
#   color=(0,255,0),
#   thickness=2
# )
# show_img(boxed_img)
# print(boxed_img.shape)



import logging
import tqdm

class TqdmLoggingHandler(logging.Handler):
  def __init__(self, level=logging.NOTSET):
    super().__init__(level)

  def emit(self, record):
    try:
      msg = self.format(record)
      tqdm.tqdm.write(msg)
      self.flush()
    except Exception:
      self.handleError(record) 