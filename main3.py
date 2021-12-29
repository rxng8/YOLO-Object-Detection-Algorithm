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
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

from yolo.const import *
from yolo.utils import draw_boxes, dynamic_iou, iou, show_img, preprocess_image
from yolo.model import SimpleModel, SimpleModel2, SimpleYolo, SimpleYolo2, testSimpleYolo
from yolo.loss import yolo_loss, simple_mse_loss

dataset_root = "./dataset/coco"
train_annotation_path = os.path.join(dataset_root, "annotations", "instances_train2014.json")
test_annotation_path = os.path.join(dataset_root, "annotations", "instances_val2014.json")
train_image_folder = os.path.join(dataset_root, "train")
test_image_folder = os.path.join(dataset_root, "val")

data_pile_path = "./dataset/coco/pickle/dump.npy"
training_history_path = "./training_history/history4.npy"
model_weights_path = "./weights/checkpoint4"

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
      preprocessed_img = preprocess_image(original_img)
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
      preprocessed_img = preprocess_image(original_img)
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
  "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_class + 5), dtype=tf.float32)
})
print(train_dataset.element_spec)
train_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_batch_iter = iter(train_batch_dataset)

test_dataset = tf.data.Dataset.from_generator(test_gen, output_signature={
  "input": tf.TensorSpec(shape=(*image_size, n_channels), dtype=tf.float32),
  "output": tf.TensorSpec(shape=(n_cell_y, n_cell_x, n_class + 5), dtype=tf.float32)
})

test_batch_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
test_batch_iter = iter(test_batch_dataset)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

model = SimpleYolo2(
  input_shape=(*image_size, n_channels),
  n_out=n_class + 5,
  n_class=n_class
)

if os.path.exists(model_weights_path + ".index"):
  print("Model weights loaded!")
  model.load_weights(model_weights_path)

# %%

## Training

def train_step(batch_x, batch_label, test_batch_iter, model, loss_function, optimizer, step=-1):
  with tf.device("/GPU:0"):
    with tf.GradientTape() as tape:
      logits = model(batch_x, training=True)
      loss = loss_function(batch_label, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

def train(model, 
        training_batch_iter, 
        test_batch_iter, 
        optimizer, 
        loss_function, 
        history, 
        epochs=1, 
        steps_per_epoch=20, 
        valid_step=5,
        history_path=None,
        weights_path=None):
  
  epochs_loss, epochs_val_loss = history
  epochs_loss = epochs_loss.tolist()
  epochs_val_loss = epochs_val_loss.tolist()

  for epoch in range(epochs):
    losses = []

    with tf.device("/CPU:0"):
      step_pointer = 0
      while step_pointer < steps_per_epoch:
        batch = next(training_batch_iter)
        batch_x = batch["input"]
        batch_label = batch["output"]
        loss = train_step(batch_x, batch_label, test_batch_iter, model, loss_function, optimizer, step=step_pointer + 1)
        print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
        losses.append(loss)

        if step_pointer % valid_step == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step_pointer + 1, float(loss))
          )
          # perform validation
          val_batch = next(test_batch_iter)
          logits = model(val_batch["input"], training=False)
          val_loss = loss_function(val_batch["output"], logits)
          print(f"Validation loss: {val_loss}\n-----------------")

        if step_pointer + 1 == steps_per_epoch:
          val_batch = next(test_batch_iter)
          logits = model(val_batch["input"], training=False)
          val_loss = loss_function(val_batch["output"], logits)
          epochs_val_loss.append(val_loss)

        step_pointer += 1
    epochs_loss.append(losses)

    # Save history and model
    if history_path != None:
      np.save(history_path, [epochs_loss, epochs_val_loss])
    
    if weights_path != None:
      model.save_weights(weights_path)
  
  # return history
  return [epochs_loss, epochs_val_loss]


# %%

if not os.path.exists(training_history_path):
  epochs_val_loss = np.array([])
  epochs_loss = np.array([])
  history = [epochs_loss, epochs_val_loss]
else:
  with open(training_history_path, "rb") as f:
    history = np.load(f, allow_pickle=True)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  yolo_loss,
  history,
  epochs=2,
  steps_per_epoch=5000, # 82783 // 16
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

confidence_threshold = 0.01

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
    print(f"Predicted class: {id_to_class[max_class_id]}")
    print(f"Confidence score: {confidence}")
    if confidence > confidence_threshold:
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


[epochs_loss, epochs_val_loss] = history
len(epochs_loss[0])
epochs_loss[0][0]


# %%

# Plot
with open(training_history_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)

# %%

e_loss = [k[0] for k in epochs_loss]

plt.plot(np.arange(1,len(e_loss)+ 1), e_loss)
plt.show()

# %%%

plt.plot(np.arange(1,len(epochs_val_loss)+ 1), epochs_val_loss)
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

loss = yolo_loss(sample_label, sample_pred)

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

lala = (intersection + 1e-9) / (union + 1e-9)

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



