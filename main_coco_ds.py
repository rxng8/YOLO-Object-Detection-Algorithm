# %%

import collections
import sys
import os
from typing import List, Dict, Tuple
import json
import csv
import pickle

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.keras.layers as L
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
%matplotlib inline
import cv2


import tqdm

from yolo.const import *
from yolo.utils import draw_boxes, dynamic_iou, iou, show_img, preprocess_image, show_img_with_bbox
from yolo.model import SimpleModel, \
  SimpleModel2, SimpleYolo, SimpleYolo2, testSimpleYolo, \
  SimpleYolo3, SimpleYolo4, make_yolov3_model
from yolo.loss import yolo_loss, simple_mse_loss, yolo_loss_2

dataset_root = "./dataset/coco"
train_annotation_path = os.path.join(dataset_root, "annotations", "instances_train2014.json")
test_annotation_path = os.path.join(dataset_root, "annotations", "instances_val2014.json")
train_image_folder = os.path.join(dataset_root, "train")
test_image_folder = os.path.join(dataset_root, "val")

data_pile_path = "./dataset/coco/pickle/dump.npy"
training_history_path = "./training_history/history9.npy"
model_weights_path = "./weights/checkpoint10"

# Data format
# https://cocodataset.org/#format-data
# images: {license, file_name, coco_url, height, width, date_captured, flickr_url, id}
# annotations: [{segmentation, area, iscrowd, image_id, bbox, category_id, id}]
# categories: [{supercategory, id, name}]

if not os.path.exists(data_pile_path):
  # Extract training
  with open(train_annotation_path) as f:
    data = json.load(f)

  # # Extract categories
  # classes: ["airplane", "bycicle", ...]
  classes: List[str] = []

  # coco_id_to_categories: {coco_id: "airplance", ...}
  coco_id_to_categories: Dict[int, str] = collections.defaultdict(str)
  for category in data["categories"]:
    coco_id_to_categories[category["id"]] = category["name"]
    classes.append(category["name"])

  # classes: ["background", "airplane", "bycicle", ...]
  classes = np.append(["background"], classes)
  n_class: int = len(classes)

  # {"background": 0, "airplane": 1, ...}
  class_to_id: Dict[str, int] = {}
  for i, _class in enumerate(classes):
    class_to_id[_class] = i

  # {0: "background", 1: "airplane", ...}
  id_to_class: Dict[int, str] = {v: k for k, v in class_to_id.items()}

  # training Data extraction: {image_id: {bbox: list, category: str}}
  train_data = collections.defaultdict(list)
  for annotation in data["annotations"]:
    x, y, width, height = annotation["bbox"]
    category_id = annotation["category_id"]
    category = coco_id_to_categories[category_id]
    image_id = annotation["image_id"]
    train_data[image_id].append({"bbox": [x, y, width, height], "category": category})

  train_data_size = len(list(train_data))

  # Process train images id
  id_to_train_image_metadata = collections.defaultdict(dict)
  for img in data["images"]:
    id_to_train_image_metadata[img["id"]]["file_name"] = img["file_name"]
    id_to_train_image_metadata[img["id"]]["width"] = img["width"]
    id_to_train_image_metadata[img["id"]]["height"] = img["height"]

  # Extract testing
  with open(test_annotation_path) as f:
    data = json.load(f)

  test_data = collections.defaultdict(list)
  for annotation in data["annotations"]:
    x, y, width, height = annotation["bbox"]
    category_id = annotation["category_id"]
    category = coco_id_to_categories[category_id]
    image_id = annotation["image_id"]
    test_data[image_id].append({"bbox": [x, y, width, height], "category": category})

  test_data_size = len(list(test_data))

  # Process train images id
  id_to_test_image_metadata = collections.defaultdict(dict)
  for img in data["images"]:
    id_to_test_image_metadata[img["id"]]["file_name"] = img["file_name"]
    id_to_test_image_metadata[img["id"]]["width"] = img["width"]
    id_to_test_image_metadata[img["id"]]["height"] = img["height"]

  np.save(data_pile_path, [
    classes, 
    class_to_id, 
    id_to_class, 
    train_data, 
    id_to_train_image_metadata,
    test_data,
    id_to_test_image_metadata
  ])

else:
  with open(data_pile_path, "rb") as f:
    [classes, \
    class_to_id, \
    id_to_class, \
    train_data, \
    id_to_train_image_metadata, \
    test_data, \
    id_to_test_image_metadata] = np.load(f, allow_pickle=True)

    n_class = len(classes)
    train_data_size = len(list(train_data))
    test_data_size = len(list(test_data))

# %%

def train_gen():
  # [(image_id, [{bbox: list, category: str}]]
  item_list = list(train_data.items())
  pointer = 0
  while True:
    if pointer >= len(item_list):
      pointer = 0
    try:
      ### Perform generator here
      image_id, item_data = item_list[pointer]
      # print(id_to_train_image_metadata[image_id])
      
      image_name = id_to_train_image_metadata[image_id]["file_name"]
      image_path = os.path.join(train_image_folder, image_name)
      original_img = np.asarray(Image.open(image_path))
      original_width = id_to_train_image_metadata[image_id]["width"]
      original_height = id_to_train_image_metadata[image_id]["height"]
      preprocessed_img = preprocess_image(original_img, image_size=example_image_size)
      # show_img(preprocessed_img)

      label = np.zeros(
        shape=(n_cell_y, n_cell_x, n_class + 5),
        dtype=float
      )

      for box_data in item_data:
        original_x, original_y, original_box_width, original_box_height = box_data["bbox"]
        _class = box_data["category"]
        _class_id = class_to_id[_class]

        # Compute regarding to the currnet image. All positions range [0, 1]
        x = float(original_x) / original_width
        y = float(original_y) / original_height
        box_width = float(original_box_width) / original_width
        box_height = float(original_box_height) / original_height
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        # print(f"x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}, center_x: {center_x}, center_y: {center_y}")

        # compute the coordinates center with regard to the current cell
        xth_cell = int(center_x * n_cell_x)
        yth_cell = int(center_y * n_cell_y)
        cell_center_x = center_x * n_cell_x - xth_cell
        cell_center_y = center_y * n_cell_y - yth_cell
        cell_box_width = box_width * n_cell_x
        cell_box_height = box_height * n_cell_y

        if label[yth_cell, xth_cell, n_class + 4] == 0:
          label[yth_cell, xth_cell, _class_id] = 1.0
          label[yth_cell, xth_cell, n_class: n_class + 4] = cell_center_x, cell_center_y, cell_box_width, cell_box_height
          label[yth_cell, xth_cell, n_class + 4] = 1.0
        # boxed = draw_boxes(preprocessed_img, [[x, y, box_width, box_height]])
        # print(f"class: {_class}")
        # show_img(boxed)

      yield {
        "input": preprocessed_img,
        "output": tf.convert_to_tensor(label)
      }

      ### End of generator performance
      pointer += 1
    except:
      pointer += 1
      continue

def test_gen():
  # [(image_id, [{bbox: list, category: str}]]
  item_list = list(test_data.items())
  pointer = 0
  while True:
    if pointer >= len(item_list):
      pointer = 0
    try:
      ### Perform generator here
      image_id, item_data = item_list[pointer]
      image_name = id_to_test_image_metadata[image_id]["file_name"]
      image_path = os.path.join(test_image_folder, image_name)
      original_img = np.asarray(Image.open(image_path))
      original_width = id_to_test_image_metadata[image_id]["width"]
      original_height = id_to_test_image_metadata[image_id]["height"]
      preprocessed_img = preprocess_image(original_img, image_size=example_image_size)
      label = np.zeros(
        shape=(n_cell_y, n_cell_x, n_class + 5),
        dtype=float
      )
      for box_data in item_data:
        original_x, original_y, original_box_width, original_box_height = box_data["bbox"]
        _class = box_data["category"]
        _class_id = class_to_id[_class]
        # Compute regarding to the currnet image. All positions range [0, 1]
        x = float(original_x) / original_width
        y = float(original_y) / original_height
        box_width = float(original_box_width) / original_width
        box_height = float(original_box_height) / original_height
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        # compute the coordinates center with regard to the current cell
        xth_cell = int(center_x * n_cell_x)
        yth_cell = int(center_y * n_cell_y)
        cell_center_x = center_x * n_cell_x - xth_cell
        cell_center_y = center_y * n_cell_y - yth_cell
        cell_box_width = box_width * n_cell_x
        cell_box_height = box_height * n_cell_y
        if label[yth_cell, xth_cell, n_class + 4] == 0:
          label[yth_cell, xth_cell, _class_id] = 1.0
          label[yth_cell, xth_cell, n_class: n_class + 4] = cell_center_x, cell_center_y, cell_box_width, cell_box_height
          label[yth_cell, xth_cell, n_class + 4] = 1.0
      yield {
        "input": preprocessed_img,
        "output": tf.convert_to_tensor(label)
      }
      ### End of generator performance
      pointer += 1
    except:
      pointer += 1
      continue

def datagen_testing(train_gen, n_iter=1):
  train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature={
    "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
    "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_class + 5), dtype=tf.float32)
  })
  print(train_dataset.element_spec)
  train_iter = iter(train_dataset)
  for i in range(n_iter):
    data = next(train_iter)
    inp = data["input"]
    outp = data["output"]
    print(f"inp shape: {inp.shape}, outp shape: {outp.shape}")

# datagen_testing(train_gen, n_iter=2)

# %%

train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature={
  "input": tf.TensorSpec(shape=(*example_image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_class + 5), dtype=tf.float32)
})
print(train_dataset.element_spec)
train_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_batch_iter = iter(train_batch_dataset)

test_dataset = tf.data.Dataset.from_generator(test_gen, output_signature={
  "input": tf.TensorSpec(shape=(*example_image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_class + 5), dtype=tf.float32)
})
test_dataset_iter = iter(test_dataset)
test_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
test_batch_iter = iter(test_batch_dataset)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# %%

# model = SimpleYolo2(
#   input_shape=(*image_size, n_channels),
#   n_out=n_class + 5,
#   n_class=n_class
# )

# model: tf.keras.Model = SimpleYolo4((*image_size, n_channels), n_class + 5, n_class) # 17, 17
model: tf.keras.Model = make_yolov3_model()
model.summary()
with tf.device("/CPU:0"):
  test_logits = tf.random.normal((BATCH_SIZE, *example_image_size, n_channels), mean=0.5, stddev=0.3)
  test_pred = model(test_logits, training=False)
  print(test_pred.shape)

# %%

## Training

def train_step(batch_x, batch_label, model, loss_function, optimizer, debug=False):
  with tf.device("/GPU:0"):
    with tf.GradientTape() as tape:
      logits = model(batch_x, training=True)
      loss, [loss_xy, loss_wh, loss_conf, loss_class] = loss_function(batch_label, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
  if debug:
    return logits, loss, [loss_xy, loss_wh, loss_conf, loss_class]
  return loss, [loss_xy, loss_wh, loss_conf, loss_class]

def train(model, 
        training_batch_iter, 
        test_batch_iter, 
        optimizer, 
        loss_function,
        epochs=1, 
        steps_per_epoch=20, 
        valid_step=5,
        history_path=None,
        weights_path=None):
  
  if not os.path.exists(training_history_path):
    epochs_val_loss = np.array([])
    epochs_loss = np.array([])
    history = [epochs_loss, epochs_val_loss]
  else:
    with open(training_history_path, "rb") as f:
      history = np.load(f, allow_pickle=True)

  epochs_loss, epochs_val_loss = history
  epochs_loss = epochs_loss.tolist()
  epochs_val_loss = epochs_val_loss.tolist()

  if os.path.exists(model_weights_path + ".index"):
    try:
      model.load_weights(model_weights_path)
      print("Model weights loaded!")
    except:
      print("Cannot load weights")

  # https://philipplies.medium.com/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5
  # outer_tqdm = tqdm(total=epochs, desc='Epoch', position=0)
  loss_logging = tqdm.tqdm(total=0, bar_format='{desc}', position=1)
  
  for epoch in range(epochs):
    losses = []
    val_losses = []

    # https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
    # https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/
    with tqdm.tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", position=0, ncols=100, ascii =".>") as inner_tqdm:
      with tf.device("/CPU:0"):
        for step_pointer in range(steps_per_epoch):
          batch = next(training_batch_iter)
          batch_x = batch["input"]
          batch_label = batch["output"]
          loss, [loss_xy, loss_wh, loss_conf, loss_class] = train_step(
            batch_x, batch_label, 
            model, loss_function, optimizer)

          # Log?
          desc = f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}"
          # loss_logging.set_description_str(desc)
          # print()

          losses.append((loss, [loss_xy, loss_wh, loss_conf, loss_class]))

          if (step_pointer + 1) % valid_step == 0:
            # desc = f"Training loss (for one batch) at step {step_pointer + 1}: {float(loss)}"
            # print(desc)
            # loss_logging.set_description_str(desc)

            # perform validation
            val_batch = next(test_batch_iter)
            logits = model(val_batch["input"], training=False)
            val_loss, [val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class] = loss_function(val_batch["output"], logits)
            val_losses.append((val_loss, [val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class]))
            # print(f"Validation loss: {val_loss}\n-----------------")

          inner_tqdm.set_postfix_str(f"Loss: {loss}")
          inner_tqdm.update(1)

    epochs_loss.append(losses)
    epochs_val_loss.append(val_losses)

    # Save history and model
    if history_path != None:
      np.save(history_path, [epochs_loss, epochs_val_loss])
    
    if weights_path != None:
      model.save_weights(weights_path)

    # outer_tqdm.update(1)

  # return history
  return [epochs_loss, epochs_val_loss]

# %%


# Debug training

def yolo_loss_3(y_true, y_pred):
  # reference: https://mlblr.com/includes/mlai/index.html#yolov2
  # reference: https://blog.emmanuelcaradec.com/humble-yolo-implementation-in-keras/
  # (batch_size, n_box_y, n_box_x, n_anchor, n_out)
  
  mask_shape = tf.shape(y_true)[:-1]

  pred_box_xy = y_pred[..., -5:-3]
  pred_box_wh = y_pred[..., -3:-1] # Adjust prediction
  pred_box_conf = y_pred[..., -1]
  pred_box_class = y_pred[..., :-5]

  true_box_xy = y_true[..., -5:-3]
  true_box_wh = y_true[..., -3:-1]
  true_box_conf = y_true[..., -1] # Shape (batch, 20, 20)
  true_box_class = y_true[..., :-5]

  # The 1_ij of the object (the ground truth of the resonsible box)
  coord_mask = y_true[..., -1] == 1.0 # Shape (batch, 20, 20, 1)

  # conf_mask
  conf_object_mask = y_true[..., -1] == 1.0

  # conf_mask
  conf_no_object_mask = y_true[..., -1] == 0.0 

  # class mask
  class_mask = y_true[..., -1] == 1.0 # Shape (batch, 20, 20)

  # Adjust the label confidence by multiplying the labeled confidence with the actual iou after predicted
  ious = dynamic_iou(y_true[..., -5:-1], y_pred[..., -5:-1]) # Shape (batch, 20, 20)
  true_box_conf = true_box_conf * ious # Shape (batch, 20, 20) x (batch, 20, 20) = (batch, 20, 20)

  # conf mask
  conf_low_conf_mask = ious < CONFIDENCE_THHRESHOLD
  conf_noobj_mask = tf.logical_and(conf_no_object_mask, conf_low_conf_mask)

  # Finalize the loss

  # compute the number of position that we are actually backpropagating
  nb_coord_box = tf.reduce_sum(tf.cast(coord_mask, dtype=tf.float32))
  nb_conf_box  = tf.reduce_sum(tf.cast(tf.logical_or(conf_object_mask, conf_noobj_mask), dtype=tf.float32))
  nb_class_box = tf.reduce_sum(tf.cast(class_mask, dtype=tf.float32))

  # Loss xy
  loss_xy = tf.reduce_sum(
    tf.square(
      true_box_xy[coord_mask] - pred_box_xy[coord_mask]
    ) * LAMBDA_COORD
  ) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse
  
  # Loss wh
  true_sqrt_box_wh = tf.sign(true_box_wh) * tf.sqrt(tf.abs(true_box_wh) + EPSILON)
  pred_sqrt_box_wh = tf.sign(pred_box_wh) * tf.sqrt(tf.abs(pred_box_wh) + EPSILON)
  loss_wh = tf.reduce_sum(
    tf.square(
      true_sqrt_box_wh[coord_mask] - pred_sqrt_box_wh[coord_mask]
    ) * LAMBDA_WH
  ) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse
  
  # Loss conf
  loss_conf_obj = tf.reduce_sum(
    tf.square(
      true_box_conf[conf_object_mask] - pred_box_conf[conf_object_mask]
    ) * LAMBDA_OBJ
  ) / (nb_conf_box + EPSILON) / 2.

  loss_conf_noobj = tf.reduce_sum(
    tf.square(
      true_box_conf[conf_noobj_mask] - pred_box_conf[conf_noobj_mask]
    ) * LAMBDA_NOOBJ
  ) / (nb_conf_box + EPSILON) / 2.
  
  loss_conf = loss_conf_obj + loss_conf_noobj

  # Loss class
  loss_class = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(
      true_box_class[class_mask], pred_box_class[class_mask], axis=-1
    ) * LAMBDA_CLASS
  ) / nb_class_box

  loss = loss_xy + loss_wh + loss_conf + loss_class
  return loss, [loss_xy, loss_wh, loss_conf, loss_class]

loop=1
test_batch_id=0
verbose=True
training = False

# Load weights if needed:
if os.path.exists(model_weights_path + ".index"):
  try:
    model.load_weights(model_weights_path)
    print("Model weights loaded!")
  except:
    print("Cannot load weights")

# Test yolo_loss_2
for i in range(loop):
  sample = next(test_batch_iter)
  sample_input = sample["input"]
  sample_output = sample["output"]
  # print(f" Sample_true shape: {sample["output"].shape}")
  if verbose:
    # show_img(sample_input[test_batch_id])
    show_img_with_bbox(sample_input, sample_output, id_to_class, 
      test_batch_id, confidence_score=0.5, display_cell=True)

  # Train
  if training:
    logits, loss, [loss_xy, loss_wh, loss_conf, loss_class] = train_step(
                sample_input, sample_output, 
                model, yolo_loss_3, optimizer, debug=True)
    print(f"loss: {loss}, loss_xy: {loss_xy}, loss_wh: {loss_wh}, loss_conf: {loss_conf}, loss_class: {loss_class}")
  else:
    logits = model(sample_input, training=True)

  if verbose:
    show_img_with_bbox(sample_input, logits, id_to_class, 
      test_batch_id, confidence_score=0.4, display_label=True, display_cell=False)


# %%

show_img_with_bbox(sample_input, logits, id_to_class, 
      test_batch_id, confidence_score=0.5, display_label=True, display_cell=False)

# %%

y_true, y_pred = sample_output, model(sample_input, training=True)
mask_shape = tf.shape(y_true)[:-1]

pred_box_xy = y_pred[..., -5:-3]
pred_box_wh = y_pred[..., -3:-1] # Adjust prediction
pred_box_conf = y_pred[..., -1]
pred_box_class = y_pred[..., :-5]

true_box_xy = y_true[..., -5:-3]
true_box_wh = y_true[..., -3:-1]
true_box_conf = y_true[..., -1] # Shape (batch, 20, 20)
true_box_class = y_true[..., :-5]

# The 1_ij of the object (the ground truth of the resonsible box)
coord_mask = y_true[..., -1] == 1.0 # Shape (batch, 20, 20, 1)

# conf_mask
conf_object_mask = y_true[..., -1] == 1.0

# conf_mask
conf_no_object_mask = y_true[..., -1] == 0.0 

# class mask
class_mask = y_true[..., -1] == 1.0 # Shape (batch, 20, 20)

# Adjust the label confidence by multiplying the labeled confidence with the actual iou after predicted
ious = dynamic_iou(y_true[..., -5:-1], y_pred[..., -5:-1]) # Shape (batch, 20, 20)
true_box_conf = true_box_conf * ious # Shape (batch, 20, 20) x (batch, 20, 20) = (batch, 20, 20)

# conf mask
conf_low_conf_mask = ious < CONFIDENCE_THHRESHOLD
conf_noobj_mask = tf.logical_and(conf_no_object_mask, conf_low_conf_mask)

# Finalize the loss

# compute the number of position that we are actually backpropagating
nb_coord_box = tf.reduce_sum(tf.cast(coord_mask, dtype=tf.float32))
nb_conf_box  = tf.reduce_sum(tf.cast(tf.logical_or(conf_object_mask, conf_noobj_mask), dtype=tf.float32))
nb_class_box = tf.reduce_sum(tf.cast(class_mask, dtype=tf.float32))

# Loss xy
loss_xy = tf.reduce_sum(
  tf.square(
    true_box_xy[coord_mask] - pred_box_xy[coord_mask]
  ) * LAMBDA_COORD
) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse

# Loss wh
true_sqrt_box_wh = tf.sign(true_box_wh) * tf.sqrt(tf.abs(true_box_wh) + EPSILON)
pred_sqrt_box_wh = tf.sign(pred_box_wh) * tf.sqrt(tf.abs(pred_box_wh) + EPSILON)
loss_wh = tf.reduce_sum(
  tf.square(
    true_sqrt_box_wh[coord_mask] - pred_sqrt_box_wh[coord_mask]
  ) * LAMBDA_WH
) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse

# Loss conf
loss_conf_obj = tf.reduce_sum(
  tf.square(
    true_box_conf[conf_object_mask] - pred_box_conf[conf_object_mask]
  ) * LAMBDA_OBJ
) / (nb_conf_box + EPSILON) / 2.

loss_conf_noobj = tf.reduce_sum(
  tf.square(
    true_box_conf[conf_noobj_mask] - pred_box_conf[conf_noobj_mask]
  ) * LAMBDA_NOOBJ
) / (nb_conf_box + EPSILON) / 2.

loss_conf = loss_conf_obj + loss_conf_noobj

# Loss class
loss_class = tf.reduce_sum(
  tf.nn.softmax_cross_entropy_with_logits(
    true_box_class[class_mask], pred_box_class[class_mask], axis=-1
  ) * LAMBDA_CLASS
) / nb_class_box

loss = loss_xy + loss_wh + loss_conf + loss_class


# %%%

# Example COCO:
import struct
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])  

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))     
    
    def reset(self):
        self.offset = 0

net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
weights_path = "./yolov3.weights"
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# make the yolov3 model to predict 80 classes on COCO
yolov3 = make_yolov3_model()

# load the weights trained on COCO into the model
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(yolov3)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                
        if label >= 0:
            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
            cv2.putText(image, 
                        label_str + ' ' + str(box.get_score()), 
                        (box.xmin, box.ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,255,0), 2)
        
    return image      


boxes = []

for i in range(len(logits)):
    # decode the output of the network
    boxes += decode_netout(logits[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

# correct the sizes of the bounding boxes
correct_yolo_boxes(boxes, net_h, net_w, net_h, net_w)

# suppress non-maximal boxes
do_nms(boxes, nms_thresh)


# %%

len(boxes)
# %%

boxes[346].ymax

# %%

a = logits[0][0]

# %%

len(logits)



# %%

# Vidualize intermediate layers

layer_list = [l for l in model.layers]
debugging_model = tf.keras.Model(model.inputs, [l.output for l in layer_list])

print(len(layer_list))
layer_list

# %%

sample = next(test_batch_iter)
sample_x = sample["input"]
sample_y_true = sample["output"]

TEST_BATCH_ID = 0
y_pred_list = debugging_model(sample_x[TEST_BATCH_ID:TEST_BATCH_ID+1], training=True)


show_img_with_bbox(sample_x, 
      y_pred_list[-1], id_to_class, 0, confidence_score=0.5, display_label=True)

# %%

f, axarr = plt.subplots(9,12, figsize=(25,15))
CONVOLUTION_NUMBER_LIST = [2, 4, 10, 15, 24, 28, 32, 36, 44, 54, 76, 85]
LAYER_LIST = [2, 35, 64, 85, 122, 145, 194, 247, 248]

for x, CONVOLUTION_NUMBER in enumerate(CONVOLUTION_NUMBER_LIST):
  try:
    f1 = y_pred_list[LAYER_LIST[0]]
    axarr[0,x].imshow(f1[0, ..., CONVOLUTION_NUMBER])
    axarr[0,x].grid(False)
  except:
    pass

  try:
    f2 = y_pred_list[LAYER_LIST[1]]
    axarr[1,x].imshow(f2[0, ..., CONVOLUTION_NUMBER])
    axarr[1,x].grid(False)
  except:
    pass

  try:
    f3 = y_pred_list[LAYER_LIST[2]]
    axarr[2,x].imshow(f3[0, ..., CONVOLUTION_NUMBER])
    axarr[2,x].grid(False)
  except:
    pass

  try:
    f4 = y_pred_list[LAYER_LIST[3]]
    axarr[3,x].imshow(f4[0, ..., CONVOLUTION_NUMBER])
    axarr[3,x].grid(False)
  except:
    pass

  try:
    f5 = y_pred_list[LAYER_LIST[4]]
    axarr[4,x].imshow(f5[0, ..., CONVOLUTION_NUMBER])
    axarr[4,x].grid(False)
  except:
    pass

  try:
    f6 = y_pred_list[LAYER_LIST[5]]
    axarr[5,x].imshow(f6[0, ..., CONVOLUTION_NUMBER])
    axarr[5,x].grid(False)
  except:
    pass

  # f7 = y_pred_list[LAYER_LIST[6]]
  # axarr[6,x].imshow(f7[TEST_BATCH_ID, ..., 85])
  # axarr[6,x].grid(False)
  
  try:
    f7 = y_pred_list[LAYER_LIST[6]]
    axarr[6,x].imshow(f7[0, ..., CONVOLUTION_NUMBER])
    axarr[6,x].grid(False)
  except:
    pass

  try:
    f8 = y_pred_list[LAYER_LIST[7]]
    axarr[7,x].imshow(f8[0, ..., CONVOLUTION_NUMBER])
    axarr[7,x].grid(False)
  except:
    pass

  try:
    f9 = y_pred_list[LAYER_LIST[8]]
    axarr[8,x].imshow(f9[0, ..., CONVOLUTION_NUMBER])
    axarr[8,x].grid(False)
  except:
    pass

# axarr[0,0].set_ylabel("After convolution layer 1")
# axarr[1,0].set_ylabel("After convolution layer 2")
# axarr[2,0].set_ylabel("After convolution layer 3")
# axarr[3,0].set_ylabel("After convolution layer 7")

# axarr[0,0].set_title("convolution number 0")
# axarr[0,1].set_title("convolution number 4")
# axarr[0,2].set_title("convolution number 7")
# axarr[0,3].set_title("convolution number 23")

plt.show()


# %%


########### TRAINING #############

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  yolo_loss_3,
  epochs=2,
  steps_per_epoch=20000, # 82783 // 4
  history_path=training_history_path,
  weights_path=model_weights_path
)

# TODO: History integration. DONE!
# TODO: mAP metrics
# TODO: Tensorboard integration
# TODO: TQDM integration. Also, time per training step/epoch
# TODO: checkpoint (save weights) integration. Including handling keyboard exception.
# TODO: lr scheduler integration


# %%

# MAnually save weights and history!
# model.save_weights(model_weights_path)
# np.save(training_history_path, history)

# %%

############## EVALUATION ##############

val_batch = next(train_batch_iter)
logits = model(val_batch["input"], training=False)
val_loss, [lxy, lwh, lconf, lcls] = yolo_loss_2(val_batch["output"], logits)

print(f"Validation loss: {val_loss}, loss xy: {lxy}, loss width height: {lwh}, loss conf: {lconf}, loss class: {lcls}")


sample_id = 0
sample_img = val_batch["input"][sample_id]
sample_label = val_batch["output"][sample_id]
sample_pred = logits[sample_id]

show_img(sample_img)
print(sample_label.shape)
assert sample_label.shape == sample_pred.shape

confidence_threshold = 0.49
boxed_img = tf.convert_to_tensor(sample_img)
for yth_cell in range(sample_pred.shape[0]):
  for xth_cell in range(sample_pred.shape[1]):
    # Draw all boxes with confidence > 0.5
    cell_center_x = sample_pred[yth_cell, xth_cell, n_class + 0] # range [0, 1]
    cell_center_y = sample_pred[yth_cell, xth_cell, n_class + 1] # range [0, 1]
    cell_bbox_width = sample_pred[yth_cell, xth_cell, n_class + 2] # range [0, 1]
    cell_bbox_height = sample_pred[yth_cell, xth_cell, n_class + 3] # range [0, 1]
    
    # Draw out!
    center_x = (cell_center_x + xth_cell) / n_cell_x
    center_y = (cell_center_y + yth_cell) / n_cell_y
    bbox_width = cell_bbox_width / n_cell_x
    bbox_height = cell_bbox_height / n_cell_y
    
    resized_xmin = center_x - bbox_width / 2.0
    resized_xmax = center_x + bbox_width / 2.0
    resized_ymin = center_y - bbox_height / 2.0
    resized_ymax = center_y + bbox_height / 2.0
    confidence = sample_pred[yth_cell, xth_cell, n_class + 4]
    pred_classes = sample_pred[yth_cell, xth_cell, 0:n_class]
    max_class_id = int(tf.argmax(pred_classes).numpy())
    # print(f"Predicted class: {id_to_class[max_class_id]}")
    # print(f"Confidence score: {confidence}")
    if confidence > confidence_threshold:
      print(f"Predicted class: {id_to_class[max_class_id]}")
      print(f"Confidence score: {confidence}")
      
      # Show the img with bounding boxes for the resized img
      boxed_img = draw_boxes(sample_img, [[resized_xmin, resized_ymin, bbox_width, bbox_height]])
      show_img(boxed_img)

# %%


# %%5

a = tf.convert_to_tensor([[2], [1]])
b = tf.convert_to_tensor([[1,2,4], [2,3,4]])

b*a

# %%5

classes

# %%

# Plot
with open(training_history_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)

flatten_epochs_loss = []
for i, epoch in enumerate(epochs_loss):
  for j, step in enumerate(epoch):
    loss, [loss_xy, loss_wh, loss_conf, loss_class] = step
    flatten_epochs_loss.append(loss)


flatten_epochs_val_loss = []
for i, epoch in enumerate(epochs_val_loss):
  # print(epoch)
  for j, step in enumerate(epoch):
    loss, [loss_xy, loss_wh, loss_conf, loss_class] = step
    flatten_epochs_val_loss.append(loss)

# compute step
val_step = len(flatten_epochs_loss) // len(flatten_epochs_val_loss)

plt.plot(np.arange(1,len(flatten_epochs_loss) + 1), flatten_epochs_loss)
plt.plot(np.arange(1,len(flatten_epochs_loss) + 1, val_step), flatten_epochs_val_loss)
plt.show()

# %%%5

epochs_val_loss

# %%

a = np.empty((1,1))
np.append(a, [[2]], axis=0)

a = a.tolist()
a

# %%

np.array([])

# %%

sample_data = next(train_batch_iter)
# %%

sample_img = sample_data["input"]
sample_label = sample_data["output"]
sample_id = 1


# %%
show_img(sample_img[sample_id])
# %%

cell_list = []

confidence_score = 0.5
for yth_cell in range(sample_label[sample_id].shape[0]):
  for xth_cell in range(sample_label[sample_id].shape[1]):
    # Draw all boxes with confidence > 0.5
    cell_center_x = sample_label[sample_id][yth_cell, xth_cell, n_class + 0] # range [0, 1]
    cell_center_y = sample_label[sample_id][yth_cell, xth_cell, n_class + 1] # range [0, 1]
    cell_bbox_width = sample_label[sample_id][yth_cell, xth_cell, n_class + 2] # range [0, 1]
    cell_bbox_height = sample_label[sample_id][yth_cell, xth_cell, n_class + 3] # range [0, 1]
    
    # Draw out!
    center_x = (cell_center_x + xth_cell) / n_cell_x
    center_y = (cell_center_y + yth_cell) / n_cell_y
    bbox_width = cell_bbox_width / n_cell_x
    bbox_height = cell_bbox_height / n_cell_y
    
    resized_xmin = center_x - bbox_width / 2.0
    resized_xmax = center_x + bbox_width / 2.0
    resized_ymin = center_y - bbox_height / 2.0
    resized_ymax = center_y + bbox_height / 2.0
    confidence = sample_label[sample_id][yth_cell, xth_cell, n_class + 4]
    pred_classes = sample_label[sample_id][yth_cell, xth_cell, 0:n_class]
    max_class_id = int(tf.argmax(pred_classes).numpy())
    if confidence > confidence_score:
      cell_list.append((yth_cell, xth_cell))
      print(f"Labeled class: {id_to_class[max_class_id]}")
      print(f"Confidence score: {confidence}")
      # Show the img with bounding boxes for the resized img
      boxed_img = draw_boxes(sample_img[sample_id], [[resized_xmin, resized_ymin, bbox_width, bbox_height]])
      show_img(boxed_img)

# %%

cell_list

# %%

sample_pred = model(sample_img)


# %%

sample_pred_item = sample_pred[sample_id]

# %%

sample_label_item = sample_label[sample_id]

# %%

loss, [lxy, lwh, lconf, lcls] = yolo_loss_2(sample_label, sample_pred)

print(f"loss: {loss}, loss xy: {lxy}, loss width height: {lwh}, loss conf: {lconf}, loss class: {lcls}")

# %%

# %%


identity_obj = sample_label[..., -1].numpy() # (batch, 20, 20)
# Shape (batch, 20, 20, n_classes + 5)


# %%

a = tf.square(label_conf * identity_obj)

# %%

sample_a = a[sample_id]

# %%
A = sample_label[..., -5:-1]
B = sample_label[..., -5:-1]

x1_A = A[..., 0:1] - (A[..., 2:3] / 2.0)
y1_A = A[..., 1:2] - (A[..., 3:4] / 2.0)
x2_A = A[..., 0:1] + (A[..., 2:3] / 2.0)
y2_A = A[..., 1:2] + (A[..., 3:4] / 2.0)

x1_B = B[..., 0:1] - (B[..., 2:3] / 2.0)
y1_B = B[..., 1:2] - (B[..., 3:4] / 2.0)
x2_B = B[..., 0:1] + (B[..., 2:3] / 2.0)
y2_B = B[..., 1:2] + (B[..., 3:4] / 2.0)

# shape (batch_size, n_cell_y, n_cell_x)
min_x2 = tf.reduce_min(tf.concat([x2_A, x2_B], axis=-1), axis=-1)
max_x1 = tf.reduce_max(tf.concat([x1_A, x1_B], axis=-1), axis=-1)
min_y2 = tf.reduce_min(tf.concat([y2_A, y2_B], axis=-1), axis=-1)
max_y1 = tf.reduce_max(tf.concat([y1_A, y1_B], axis=-1), axis=-1)

# shape (batch_size, n_cell_y, n_cell_x)
intersection = tf.math.maximum(0, min_x2 - max_x1) * tf.math.maximum(0, min_y2 - max_y1)

# (batch_size, n_cell_y, n_cell_x)
union = tf.squeeze((x2_A - x1_A) * (y2_A - y1_A) + (x2_B - x1_B) * (y2_B - y1_B), axis=-1) - intersection

lala = (intersection + EPSILON) / (union + EPSILON)

# %%

ious = dynamic_iou(sample_label[..., -5:-1], sample_label[..., -5:-1]) # Shape (batch, 20,20)
# ious = ious[..., tf.newaxis] # (batch, 20, 20, 1)
label_conf = sample_label[..., -1] * lala



# %%

ious = dynamic_iou(sample_label[..., -5:-1], sample_pred[..., -5:-1]) # Shape (batch, 20,20)
# ious = ious[..., tf.newaxis] # (batch, 20, 20, 1)
label_conf = sample_label[..., -1] * ious

# Coordinates x, y loss
# loss_xy shape (batch, 20, 20)
loss_xy = LAMBDA_COORD * tf.reduce_sum(tf.square(sample_label[..., -5:-3]*tf.expand_dims(identity_obj,-1) - sample_pred[..., -5:-3]*tf.expand_dims(identity_obj,-1)), axis=-1)
# loss_wh shape (batch, 20, 20)
loss_wh = LAMBDA_COORD * tf.reduce_sum(tf.square(tf.sign(sample_label[..., -3:-1])*tf.sqrt(tf.abs(sample_label[..., -3:-1]) + 1e-6)*tf.expand_dims(identity_obj,-1) - tf.sign(sample_pred[..., -3:-1])*tf.sqrt(tf.abs(sample_pred[..., -3:-1]) + 1e-6)*tf.expand_dims(identity_obj,-1)), axis=-1) 
# loss_class shape (batch, 20, 20)
loss_class = tf.reduce_sum(tf.square(sample_label[..., :-5] - sample_pred[..., :-5]) * tf.expand_dims(identity_obj, -1), axis=-1)

# loss_conf shape (batch, 20, 20)
loss_conf = tf.square(label_conf * identity_obj - sample_pred[..., -1] * identity_obj) \
  + LAMBDA_NOOBJ * tf.square(label_conf * (1 - identity_obj) - sample_pred[..., -1] * (1 - identity_obj))

# element wise addition
loss = (loss_xy + loss_wh + loss_class + loss_conf)



# %%%


a = tf.random.normal((2, 3, 3, 6)) # (batch, size_x, size_y, vector)
b = np.zeros((2,3,3,6))
b[1, 2, 2, 5] = 1
b[0, 1, 0, 5] = 1
mask = b[..., -1] > 0.5

# %%

mask

# %%

a[mask]




