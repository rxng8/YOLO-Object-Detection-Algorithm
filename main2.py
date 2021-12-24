# %%

import collections
import sys
import os
from typing import List, Dict, Tuple
import json
import csv

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

from yolo.const import *
from yolo.utils import iou, show_img
from yolo.model import SimpleModel, SimpleModel2, SimpleYolo, testSimpleYolo
from yolo.loss import yolo_loss, simple_mse_loss

dataset_root = "./dataset/card_dataset"
train_label_path = os.path.join(dataset_root, "train_labels.csv")
test_label_path = os.path.join(dataset_root, "test_labels.csv")
train_images_path = os.path.join(dataset_root, "train")
test_images_path = os.path.join(dataset_root, "test")

train_label_df = pd.read_csv(train_label_path)
test_label_df = pd.read_csv(test_label_path)

classes = train_label_df["class"].unique()
classes = np.append(["background"], classes)
n_class = len(classes)
print(f"There are {n_class} total number of classes.")
class_to_id = {}
for i, _class in enumerate(classes):
  class_to_id[_class] = i
id_to_class = {v: k for k, v in class_to_id.items()}

# Build train_label
train_label = collections.defaultdict(list)
for i, row in train_label_df.iterrows():
  train_label[row["filename"]].append({
    "class": row["class"],
    "width": int(row["width"]),
    "height": int(row["height"]),
    "xmin": int(row["xmin"]),
    "ymin": int(row["ymin"]),
    "xmax": int(row["xmax"]),
    "ymax": int(row["ymax"])
  })

test_label = collections.defaultdict(list)
for i, row in test_label_df.iterrows():
  test_label[row["filename"]].append({
    "class": row["class"],
    "width": int(row["width"]),
    "height": int(row["height"]),
    "xmin": int(row["xmin"]),
    "ymin": int(row["ymin"]),
    "xmax": int(row["xmax"]),
    "ymax": int(row["ymax"])
  })

# testSimpleYolo()

# %%

def train_datagen():
  # Preprocess the df first
  
  item_list = list(train_label.items())
  pointer = 0
  while True:
    if pointer >= len(item_list):
      pointer = 0
    filename, item_data = item_list[pointer]
    original_img = np.asarray(Image.open(os.path.join(train_images_path, filename)))
    # show_img(original_img)

    # Preprocessing image
    preprocessed_img = tf.cast(original_img, tf.float32)
    preprocessed_img /= 255.0
    assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
    preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    # show_img(preprocessed_img)

    # the label vector is (n_box_y, n_box_x, n_anchor_box, n_class + 4)
    # n_class + 5 includes the class probabilities, the bounding boxes coordinates (center_x, centeR_y, width, height), and the probability of an object existing (the predicted confidence score).
    label = np.zeros(
      shape=(n_cell_y, n_cell_x, n_anchor, n_class + 5),
      dtype=float
    )

    # Build label vector
    for box in item_data:
      resized_xmin = float(box["xmin"]) / box["width"] # range [0, 1]
      resized_ymin = float(box["ymin"]) / box["height"] # range [0, 1]
      resized_xmax = float(box["xmax"]) / box["width"] # range [0, 1]
      resized_ymax = float(box["ymax"]) / box["height"] # range [0, 1]
      bbox_width = resized_xmax - resized_xmin
      bbox_height = resized_ymax - resized_ymin
      center_x = resized_xmin + float(bbox_width) / 2
      center_y = resized_ymin + float(bbox_height) / 2

      # print(f"center_x: {center_x}, center_y: {center_y}")

      xth_cell = int(center_x * n_cell_x)
      yth_cell = int(center_y * n_cell_y)

      # print(f"xth_cell: {xth_cell}, yth_cell: {yth_cell}")

      # center x of the true bounding box but relative to the current cell
      cell_center_x = (center_x - float(xth_cell) / n_cell_x) / (1.0 / n_cell_x)
      cell_center_y = (center_y - float(yth_cell) / n_cell_y) / (1.0 / n_cell_y)

      # Width and height of the boubnding box with reagard to the cell
      cell_bbox_width = bbox_width * n_cell_x
      cell_bbox_height = bbox_height * n_cell_y

      # print(f"cell_center_x: {cell_center_x}, cell_center_y: {cell_center_y}, cell_bbox_width: {cell_bbox_width}, cell_bbox_height: {cell_bbox_height}")

      best_anchor_box_id = -1
      current_best_iou = -1
      for i, size in enumerate(anchor_size):
        for j, (height_ratio, width_ratio) in enumerate(anchor_ratio):
          anchor_box_id = len(anchor_size) * i + j # This is computed by anchor_size_th * len(anchor_size) + anchor_ratio_th
          # compute iou between the anchor box and the actual bounding box
          # compute xmin, ymin, xmax, ymax of the anchor box
          anchor_box_xmin = center_x - (float(size)/image_size[1]) / 2 * width_ratio
          anchor_box_xmax = center_x + (float(size)/image_size[1]) / 2 * width_ratio
          anchor_box_ymin = center_y - (float(size)/image_size[0]) / 2 * height_ratio
          anchor_box_ymax = center_y + (float(size)/image_size[0]) / 2 * height_ratio

          current_iou = iou(
            anchor_box_xmin,
            anchor_box_ymin,
            anchor_box_xmax,
            anchor_box_ymax,
            resized_xmin,
            resized_ymin,
            resized_xmax,
            resized_ymax
          )
          assert current_iou >= 0
          if current_iou > current_best_iou:
            best_anchor_box_id = anchor_box_id
            current_best_iou = current_iou
      
      # print(f"The best anchorbox is with size {anchor_size[best_anchor_box_id // len(anchor_size)]}, and ratio {anchor_ratio[best_anchor_box_id % len(anchor_ratio)]}")

      # Set the label to the label
      # label[yth_cell, xth_cell, best_anchor_box_id, class_to_id[box["class"]]] = 1.0
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 0] = cell_center_x
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 1] = cell_center_y
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 2] = cell_bbox_width
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 3] = cell_bbox_height
      label[yth_cell, xth_cell, best_anchor_box_id, n_class + 4] = 1.0

    # Set for every cell
    label[yth_cell, xth_cell, :, class_to_id[box["class"]]] = 1.0
    label[yth_cell, xth_cell, :, n_class + 0] = cell_center_x
    label[yth_cell, xth_cell, :, n_class + 1] = cell_center_y
    label[yth_cell, xth_cell, :, n_class + 2] = cell_bbox_width
    label[yth_cell, xth_cell, :, n_class + 3] = cell_bbox_height

    # increase the pointer for the next yield
    pointer += 1

    yield {
      "input": preprocessed_img,
      "output": tf.convert_to_tensor(label)
    }


# # %%

# # Debug
# train_dataset = tf.data.Dataset.from_generator(train_datagen, output_signature={
#   "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
#   "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_anchor, n_class + 5), dtype=tf.float32)
# })
# print(train_dataset.element_spec)
# train_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
# train_batch_iter = iter(train_dataset)

# # %%
# a = next(train_batch_iter)

# # %%

def test_datagen():
  # Preprocess the df first
  
  item_list = list(test_label.items())
  pointer = 0
  while True:
    if pointer >= len(item_list):
      pointer = 0
    filename, item_data = item_list[pointer]
    original_img = np.asarray(Image.open(os.path.join(test_images_path, filename)))
    # show_img(original_img)

    # Preprocessing image
    preprocessed_img = tf.cast(original_img, tf.float32)
    preprocessed_img /= 255.0
    assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
    preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    # show_img(preprocessed_img)

    # the label vector is (n_box_y, n_box_x, n_anchor_box, n_class + 4)
    # n_class + 5 includes the class probabilities, the bounding boxes coordinates (center_x, centeR_y, width, height), and the probability of an object existing (the predicted confidence score).
    label = np.zeros(
      shape=(n_cell_y, n_cell_x, n_anchor, n_class + 5),
      dtype=float
    )

    # Build label vector
    for box in item_data:
      resized_xmin = float(box["xmin"]) / box["width"] # range [0, 1]
      resized_ymin = float(box["ymin"]) / box["height"] # range [0, 1]
      resized_xmax = float(box["xmax"]) / box["width"] # range [0, 1]
      resized_ymax = float(box["ymax"]) / box["height"] # range [0, 1]
      bbox_width = resized_xmax - resized_xmin
      bbox_height = resized_ymax - resized_ymin
      center_x = resized_xmin + float(bbox_width) / 2
      center_y = resized_ymin + float(bbox_height) / 2

      # print(f"center_x: {center_x}, center_y: {center_y}")

      xth_cell = int(center_x * n_cell_x)
      yth_cell = int(center_y * n_cell_y)

      # print(f"xth_cell: {xth_cell}, yth_cell: {yth_cell}")

      # center x of the true bounding box but relative to the current cell
      cell_center_x = (center_x - float(xth_cell) / n_cell_x) / (1.0 / n_cell_x)
      cell_center_y = (center_y - float(yth_cell) / n_cell_y) / (1.0 / n_cell_y)

      # Width and height of the boubnding box with reagard to the cell
      cell_bbox_width = bbox_width * n_cell_x
      cell_bbox_height = bbox_height * n_cell_y

      # print(f"cell_center_x: {cell_center_x}, cell_center_y: {cell_center_y}, cell_bbox_width: {cell_bbox_width}, cell_bbox_height: {cell_bbox_height}")

      best_anchor_box_id = -1
      current_best_iou = -1
      for i, size in enumerate(anchor_size):
        for j, (height_ratio, width_ratio) in enumerate(anchor_ratio):
          anchor_box_id = len(anchor_size) * i + j # This is computed by anchor_size_th * len(anchor_size) + anchor_ratio_th
          # compute iou between the anchor box and the actual bounding box
          # compute xmin, ymin, xmax, ymax of the anchor box
          anchor_box_xmin = center_x - (float(size)/image_size[1]) / 2 * width_ratio
          anchor_box_xmax = center_x + (float(size)/image_size[1]) / 2 * width_ratio
          anchor_box_ymin = center_y - (float(size)/image_size[0]) / 2 * height_ratio
          anchor_box_ymax = center_y + (float(size)/image_size[0]) / 2 * height_ratio

          current_iou = iou(
            anchor_box_xmin,
            anchor_box_ymin,
            anchor_box_xmax,
            anchor_box_ymax,
            resized_xmin,
            resized_ymin,
            resized_xmax,
            resized_ymax
          )
          assert current_iou >= 0
          if current_iou > current_best_iou:
            best_anchor_box_id = anchor_box_id
            current_best_iou = current_iou
      
      # print(f"The best anchorbox is with size {anchor_size[best_anchor_box_id // len(anchor_size)]}, and ratio {anchor_ratio[best_anchor_box_id % len(anchor_ratio)]}")

      # Set the label to the label
      # label[yth_cell, xth_cell, best_anchor_box_id, class_to_id[box["class"]]] = 1.0
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 0] = cell_center_x
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 1] = cell_center_y
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 2] = cell_bbox_width
      # label[yth_cell, xth_cell, best_anchor_box_id, n_class + 3] = cell_bbox_height
      label[yth_cell, xth_cell, best_anchor_box_id, n_class + 4] = 1.0

    # Set for every cell
    label[yth_cell, xth_cell, :, class_to_id[box["class"]]] = 1.0
    label[yth_cell, xth_cell, :, n_class + 0] = cell_center_x
    label[yth_cell, xth_cell, :, n_class + 1] = cell_center_y
    label[yth_cell, xth_cell, :, n_class + 2] = cell_bbox_width
    label[yth_cell, xth_cell, :, n_class + 3] = cell_bbox_height

    # increase the pointer for the next yield
    pointer += 1

    yield {
      "input": preprocessed_img,
      "output": tf.convert_to_tensor(label)
    }

def train_step(batch_x, batch_label, test_batch_iter, model, loss_function, optimizer, step=-1, valid_step=5):
  batch_size = batch_x.shape[0]
  with tf.device("/GPU:0"):
    with tf.GradientTape() as tape:
      logits = model(batch_x, training=True)
      loss = loss_function(batch_label, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

  if step % valid_step == 0:
    print(
        "Training loss (for one batch) at step %d: %.4f"
        % (step, float(loss))
    )
    print("Seen so far: %s samples" % ((step + 1) * batch_size))
    # perform validation
    val_batch = next(test_batch_iter)
    logits = model(val_batch["input"], training=False)
    val_loss = loss_function(val_batch["output"], logits)
    print(f"Validation loss: {val_loss}\n-----------------")
    
  return loss

def train(model, training_batch_iter, test_batch_iter, optimizer, loss_function, epochs=1, steps_per_epoch=20):
  for epoch in range(epochs):
    with tf.device("/CPU:0"):
      step_pointer = 0
      while step_pointer < steps_per_epoch:
        batch = next(training_batch_iter)
        batch_x = batch["input"]
        batch_label = batch["output"]
        loss = train_step(batch_x, batch_label, test_batch_iter, model, loss_function, optimizer, step=step_pointer + 1)
        print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
        step_pointer += 1


# %%

train_dataset = tf.data.Dataset.from_generator(train_datagen, output_signature={
  "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_anchor, n_class + 5), dtype=tf.float32)
})
print(train_dataset.element_spec)
train_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_batch_iter = iter(train_batch_dataset)

test_dataset = tf.data.Dataset.from_generator(test_datagen, output_signature={
  "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_anchor, n_class + 5), dtype=tf.float32)
})

test_batch_dataset = train_dataset.batch(batch_size=VAL_BATCH_SIZE)
test_batch_iter = iter(test_batch_dataset)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# model = SimpleModel2(
#   input_shape=(*image_size, n_channels),
#   n_anchor_boxes=n_anchor,
#   n_out=n_class+5
# )

model = SimpleYolo(
  input_shape=(*image_size, n_channels),
  n_anchor_boxes=n_anchor,
  n_out=n_class + 5,
  n_class=n_class
)

# %%

train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  yolo_loss,
  epochs=10,
  steps_per_epoch=40 # 592 // 16
)

# %%

a = tf.convert_to_tensor([[3,4],[5,6]])

a[..., 1:]

# %%

testSimpleYolo()

# %%


model.save_weights("./weights/checkpoint")

# %%

# Test display data

val_batch = next(test_batch_iter)
logits = model(val_batch["input"], training=False)
val_loss = yolo_loss(val_batch["output"], logits)
print(f"Validation loss: {val_loss}")

sample_id = 0
sample_img = val_batch["input"][sample_id]
sample_label = val_batch["output"][sample_id]
sample_pred = logits[sample_id]

show_img(sample_img)
print(sample_label.shape)
assert sample_label.shape == sample_pred.shape

# %%

confidence_threshold = 0.2

for y in range(sample_pred.shape[0]):
  for x in range(sample_pred.shape[1]):
    for anchor_box_id in range(sample_pred.shape[2]):
      # Draw all boxes with confidence > 0.5
      cell_center_x = sample_pred[y, x, anchor_box_id, n_class + 0] # range [0, 1]
      cell_center_y = sample_pred[y, x, anchor_box_id, n_class + 1] # range [0, 1]
      cell_bbox_width = sample_pred[y, x, anchor_box_id, n_class + 2] # range [0, 1]
      cell_bbox_height = sample_pred[y, x, anchor_box_id, n_class + 3] # range [0, 1]
      
      # Draw out!
      center_x = cell_center_x / n_cell_x + float(x) / n_cell_x
      center_y = cell_center_y / n_cell_y + float(y) / n_cell_y
      bbox_width = cell_bbox_width / n_cell_x
      bbox_height = cell_bbox_height / n_cell_y
      
      resized_xmin = center_x - bbox_width / 2.0
      resized_xmax = center_x + bbox_width / 2.0
      resized_ymin = center_y - bbox_height / 2.0
      resized_ymax = center_y + bbox_height / 2.0
      confidence = sample_pred[y, x, anchor_box_id, n_class + 4]
      pred_classes = sample_pred[y, x, anchor_box_id, 0:n_class]
      print(f"Predicted class: {id_to_class[int(tf.reduce_max(pred_classes))]}")
      print(f"Confidence score: {confidence}")
      if True: # confidence > confidence_threshold:
        # Show the img with bounding boxes for the resized img
        boxed_img = cv2.rectangle(
          sample_img.numpy(),
          (int(resized_xmin * image_size[1]), int(resized_ymin * image_size[0])), # (x1, y1)
          (int(resized_xmax * image_size[1]), int(resized_ymax * image_size[0])), # (x2, y2)
          color=(0,1,0),
          thickness=1
        )
        show_img(boxed_img)


# %%

# **Visualizing Intermediate Representations**
from IPython.display import clear_output
from matplotlib import animation

# myplot = None
# plt.ion()
# figure, ax = plt.subplots(figsize=(10, 8))
# # sample_img = sample_img.numpy()
# while True:
#   figure = plt.imshow(sample_img)
#   plt.show()
#   sample_img[23: 65, 53: 90, :] += 0.2
#   if tf.reduce_max(sample_img) > 1:
#     sample_img[...] = 0
#   clear_output(wait=True)


successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict(next(test_batch_iter)["input"])
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  pass


# %%


for item in train_batch_dataset:
  # print(item["input"])
  result = model(item["input"])
  print(result)
  break

# %%

a = [[3,4,5],[5,6,7]]
a = tf.convert_to_tensor(a)
cond = a[:,2] == 7
print(cond)

cond = tf.expand_dims(cond, -1)
cond = tf.broadcast_to(cond, (*cond.shape[:-1], 3))
print(cond)


# %%

test_batch = next(iter(test_batch_dataset))
# %%

x = test_batch["input"]
y_true = test_batch["output"]

with tf.device("/CPU:0"):
  logits = model(x, training=False)
  loss = simple_mse_loss(test_batch["output"], logits)

# %%

show_img(x[0])

# Show the img with bounding boxes for the resized img
boxed_img = cv2.rectangle(
  x[0].numpy(),
  (int(resized_xmin * image_size[1]), int(resized_ymin * image_size[0])), # (x1, y1)
  (int(resized_xmax * image_size[1]), int(resized_ymax * image_size[0])), # (x2, y2)
  color=(0,255,0),
  thickness=2
)
show_img(boxed_img)
print(boxed_img.shape)


# %%


a = [[[2,3,4], [5,6,7], [4,5,7]], [[9,34,4], [2,6,9], [20,14,15]], [[13,13,13], [543,64,72], [42,53,7]]]
a = tf.convert_to_tensor(a)


cond_object = a[..., 2] == 7
cond_object = tf.expand_dims(cond_object, -1)
cond_object = tf.broadcast_to(cond_object, (*cond_object.shape[:-1], 3))
cond_object = cond_object.numpy()
cond_object[..., 2] = True
cond_object
# %%


