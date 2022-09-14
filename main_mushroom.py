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
import tensorflow.keras.layers as layers
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
from yolo.loss import yolo_loss, simple_mse_loss, yolo_loss_2, yolo_loss_3

dataset_root = "./dataset/mushroom"
train_annotation_path = os.path.join(dataset_root, "train", "_annotations.coco.json")
test_annotation_path = os.path.join(dataset_root, "test", "_annotations.coco.json")
train_image_folder = os.path.join(dataset_root, "train")
test_image_folder = os.path.join(dataset_root, "test")

data_pile_path = "./dataset/mushroom/pickle/dump.npy"
training_history_path = "./training_history/history_mushroom_1.npy"
model_weights_path = "./weights/mushhroom/checkpoint_1"

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
model: tf.keras.Model = make_yolov3_model(n_class=n_class)
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

loop=1
test_batch_id=0
verbose=True
training = False

# Load weights if needed:
# if os.path.exists(model_weights_path + ".index"):
#   try:
#     model.load_weights(model_weights_path)
#     print("Model weights loaded!")
#   except:
#     print("Cannot load weights")

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
    logits = model(sample_input, training=False)
    loss, [loss_xy, loss_wh, loss_conf, loss_class] = yolo_loss_3(sample_output, logits)
    print(f"loss: {loss}, loss_xy: {loss_xy}, loss_wh: {loss_wh}, loss_conf: {loss_conf}, loss_class: {loss_class}")

  if verbose:
    show_img_with_bbox(sample_input, logits, id_to_class, 
      test_batch_id, confidence_score=0.9, display_label=True, display_cell=False)


# show_img_with_bbox(sample_input, logits, id_to_class, 
#       test_batch_id, confidence_score=0.5, display_label=True, display_cell=False)

# %%


# def do_nms(boxes, nms_thresh):
#     if len(boxes) > 0:
#         nb_class = len(boxes[0].classes)
#     else:
#         return
        
#     for c in range(nb_class):
#         sorted_indices = np.argsort([-box.classes[c] for box in boxes])

#         for i in range(len(sorted_indices)):
#             index_i = sorted_indices[i]

#             if boxes[index_i].classes[c] == 0: continue

#             for j in range(i+1, len(sorted_indices)):
#                 index_j = sorted_indices[j]

#                 if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
#                     boxes[index_j].classes[c] = 0
                    
# def draw_boxes(image, boxes, labels, obj_thresh):
#     for box in boxes:
#         label_str = ''
#         label = -1
        
#         for i in range(len(labels)):
#             if box.classes[i] > obj_thresh:
#                 label_str += labels[i]
#                 label = i
#                 print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                
#         if label >= 0:
#             cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
#             cv2.putText(image, 
#                         label_str + ' ' + str(box.get_score()), 
#                         (box.xmin, box.ymin - 13), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 
#                         1e-3 * image.shape[0], 
#                         (0,255,0), 2)
        
#     return image      


# boxes = []

# for i in range(len(logits)):
#     # decode the output of the network
#     boxes += decode_netout(logits[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

# # correct the sizes of the bounding boxes
# correct_yolo_boxes(boxes, net_h, net_w, net_h, net_w)

# # suppress non-maximal boxes
# do_nms(boxes, nms_thresh)


# %%

# Vidualize intermediate layers
CHOSEN_LAYER = [5, 26, 44, 65, 86, 94, 100]
layer_list = [l for l in model.layers]
layer_list = [layer_list[i] for i in CHOSEN_LAYER]
layer_name_list = [l.name for l in layer_list]
debugging_model = tf.keras.Model(model.inputs, [l.output for l in layer_list])

print(len(layer_list))
layer_list

# %%

sample = next(test_batch_iter)
sample_x = sample["input"]
sample_y_true = sample["output"]

TEST_BATCH_ID = 0
# Activations
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

# GRADCAM Visualization addweights heatmap to the image

def get_CAM(model, img, actual_label, loss_func, layer_name='block5_conv3'):

  model_grad = tf.keras.Model(model.inputs, 
                      [model.get_layer(layer_name).output, model.output])
  
  with tf.GradientTape() as tape:
      conv_output_values, predictions = model_grad(img)

      # watch the conv_output_values
      tape.watch(conv_output_values)
      
      # Calculate loss as in the loss func
      try:
        loss, _ = loss_func(actual_label, predictions)
      except:
        loss = loss_func(actual_label, predictions)
      print(f"Loss: {loss}")
  
  # get the gradient of the loss with respect to the outputs of the last conv layer
  grads_values = tape.gradient(loss, conv_output_values)
  grads_values = tf.reduce_mean(grads_values, axis=(0,1,2))
  
  conv_output_values = np.squeeze(conv_output_values.numpy())
  grads_values = grads_values.numpy()
  
  # weight the convolution outputs with the computed gradients
  for i in range(conv_output_values.shape[-1]): 
      conv_output_values[:,:,i] *= grads_values[i]
  heatmap = np.mean(conv_output_values, axis=-1)
  
  heatmap = np.maximum(heatmap, 0)
  heatmap /= heatmap.max()
  
  del model_grad, conv_output_values, grads_values, loss
  
  return heatmap

activations = debugging_model(sample_x[TEST_BATCH_ID:TEST_BATCH_ID+1, ...], training=True)

inspecting_layer_index = np.random.randint(0, len(layer_list))
inspecting_layer_number = layer_list[inspecting_layer_index]
inspecting_layer_name = layer_name_list[inspecting_layer_index]

heatmap = get_CAM(model, 
  sample_x[TEST_BATCH_ID:TEST_BATCH_ID+1, ...], 
  sample_y_true[TEST_BATCH_ID:TEST_BATCH_ID+1, ...], 
  yolo_loss_3, layer_name=inspecting_layer_name)

heatmap = cv2.resize(heatmap, (example_image_size[0], example_image_size[1]))
heatmap = heatmap * 255
heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
converted_img = sample_x[TEST_BATCH_ID, ...].numpy()
super_imposed_image = cv2.addWeighted(converted_img, 0.8, heatmap.astype('float32'), 2e-3, 0.0)

f,ax = plt.subplots(2,2, figsize=(15,8))

ax[0,0].imshow(sample_x[TEST_BATCH_ID, ...])
# ax[0,0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
ax[0,0].axis('off')

conv_number = 8
sample_activation = activations[inspecting_layer_index][0,:,:,conv_number]
sample_activation-=tf.reduce_mean(sample_activation).numpy()
sample_activation/=tf.math.reduce_std(sample_activation).numpy()
sample_activation *=255
sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)
ax[0,1].imshow(sample_activation)
ax[0,1].set_title("Random feature map")
ax[0,1].axis('off')

ax[1,0].imshow(heatmap)
ax[1,0].set_title("Class Activation Map")
ax[1,0].axis('off')

ax[1,1].imshow(super_imposed_image)
ax[1,1].set_title("Activation map superimposed")
ax[1,1].axis('off')
plt.tight_layout()
plt.show()

# %%%

##### Visualize intermediary #######

def visualize_intermediate_activations(layer_names, activations):
  assert len(layer_names)==len(activations), "Make sure layers and activation values match"
  images_per_row=16
  
  for layer_name, layer_activation in zip(layer_names, activations):
    nb_features = layer_activation.shape[-1]
    size= layer_activation.shape[1]

    nb_cols = nb_features // images_per_row
    grid = np.zeros((size*nb_cols, size*images_per_row))

    for col in range(nb_cols):
      for row in range(images_per_row):
        feature_map = layer_activation[0,:,:,col*images_per_row + row]
        feature_map -= tf.reduce_mean(feature_map).numpy()
        feature_map /= tf.math.reduce_std(feature_map).numpy()
        feature_map *=255
        feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

        grid[col*size:(col+1)*size, row*size:(row+1)*size] = feature_map

    scale = 1./size
    plt.figure(figsize=(scale*grid.shape[1], scale*grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(grid, aspect='auto', cmap='viridis')
  plt.show()

visualize_intermediate_activations(layer_name_list, y_pred_list)

# %%


########### TRAINING #############

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  yolo_loss_3,
  epochs=50,
  steps_per_epoch=62, # 247 // 4
  history_path=training_history_path,
  weights_path=model_weights_path
)

# TODO: mAP metrics
# TODO: Tensorboard integration
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

######## Plot history #########

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


# %%
