import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import ImageDraw, ImageFont, Image

from .const import *

def iou(A, B):
  # xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2

  xmin1, ymin1, xmax1, ymax1 = A
  xmin2, ymin2, xmax2, ymax2 = B

  intersection = (min(xmax1, xmax2) - max(xmin1, xmin2)) * (min(ymax1, ymax2) - max(ymin1, ymin2))
  intersection = max(0, intersection)
  union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
  return float(intersection + EPSILON) / (union + EPSILON)

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
      box (List[List[float]]): Expect list of boxes, each box is [x, y, width, height]. 
        (x, y) is the top-left point. Ranged from 0 to 1.
  """
  img_height, img_width = img.shape[0:2]
  try:
    boxed_img = img.numpy()
  except:
    boxed_img = img
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



def draw_a_box(PIL_image, 
    box: List[float], 
    color="red", 
    thickness=1, 
    display_str_list: Tuple[str]=()):
  """ inspiration: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

  Args:
      img ([type]): tensorflow Tensor object
      box (List[List[float]]): Expect list of boxes, each box is [x, y, width, height]. 
        (x, y) is the top-left point. Ranged from 0 to 1.
  """
  
  draw = ImageDraw.Draw(PIL_image)
  im_width, im_height = PIL_image.size
  
  x, y, width, height = box
  xmin = int(x * im_width)
  ymin = int(y * im_height)
  xmax = int((x+width) * im_width)
  ymax = int((y+height) * im_height)
  left = xmin
  right = xmax
  top = ymin
  bottom = ymax
  if thickness > 0:
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
              (left, top)],
              width=thickness,
              fill=color)

  try:
    font = ImageFont.truetype('arial.ttf', 16)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  
  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes_2(img, 
    boxes: List[List[float]], 
    color="red", 
    thickness=1, 
    display_str_lists: List[Tuple[str]]=[]):
  
  try:
    current_img = img.numpy()
  except:
    current_img = img

  PIL_image = Image.fromarray((current_img * 255).astype('uint8'), 'RGB')

  # draw boxes
  for i in range(len(boxes)):
    display_str_list = ()
    if display_str_lists:
      try:
        display_str_list = display_str_lists[i]
      except:
        pass
    draw_a_box(PIL_image, boxes[i], color, thickness, display_str_list)

  returned_image = np.zeros(current_img.shape, dtype="uint8")
  np.copyto(returned_image, np.array(PIL_image))

  return returned_image


def show_img_with_bbox(sample_input, 
    sample_output,
    id_to_class: Dict[int, str],
    sample_batch_id=0, 
    confidence_score=0.5,
    display_label=True):
  
  cell_list = []

  boxes_list = []
  display_str_lists = []

  sample_img = sample_input[sample_batch_id]
  
  for yth_cell in range(sample_output[sample_batch_id].shape[0]):
    for xth_cell in range(sample_output[sample_batch_id].shape[1]):
      # Draw all boxes with confidence > 0.5
      cell_center_x = sample_output[sample_batch_id][yth_cell, xth_cell, -5] # range [0, 1]
      cell_center_y = sample_output[sample_batch_id][yth_cell, xth_cell, -4] # range [0, 1]
      cell_bbox_width = sample_output[sample_batch_id][yth_cell, xth_cell, -3] # range [0, ~]
      cell_bbox_height = sample_output[sample_batch_id][yth_cell, xth_cell, -2] # range [0, ~]
      
      # Draw out!
      center_x = (cell_center_x + xth_cell) / n_cell_x
      center_y = (cell_center_y + yth_cell) / n_cell_y
      bbox_width = cell_bbox_width / n_cell_x
      bbox_height = cell_bbox_height / n_cell_y
      
      resized_xmin = center_x - bbox_width / 2.0
      resized_xmax = center_x + bbox_width / 2.0
      resized_ymin = center_y - bbox_height / 2.0
      resized_ymax = center_y + bbox_height / 2.0
      confidence = sample_output[sample_batch_id][yth_cell, xth_cell, -1]
      
      if confidence > confidence_score:
        cell_list.append((yth_cell, xth_cell))
        # print(f"Labeled class: {id_to_class[max_class_id]}")
        # print(f"Confidence score: {confidence}")
        # Show the img with bounding boxes for the resized img
        pred_classes = sample_output[sample_batch_id][yth_cell, xth_cell, :-5]
        max_class_id = int(tf.argmax(pred_classes).numpy())

        boxes_list.append([resized_xmin, resized_ymin, bbox_width, bbox_height])
        display_str_lists.append([f"{id_to_class[max_class_id]}"])
  
  boxed_img = sample_img.numpy()
  boxed_img = draw_boxes_2(boxed_img, boxes_list, display_str_lists=display_str_lists if display_label else ())
  show_img(boxed_img)

