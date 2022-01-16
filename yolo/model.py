from types import new_class
import tensorflow as tf
import tensorflow.keras.layers as layers

from .const import *

# Write model
class SimpleModel(tf.keras.Model):
  def __init__(self, input_shape, n_anchor_boxes, n_out) -> None:
    super().__init__()
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv2D(n_anchor_boxes * n_out, (1,1), padding="same", activation="sigmoid"),
      tf.keras.layers.Reshape((20, 20, n_anchor_boxes, n_out))
    ])
    self.model.summary()

  def call(self, x):
    return self.model(x)

class SimpleModel2(tf.keras.Model):
  def __init__(self, input_shape, n_anchor_boxes, n_out) -> None:
    super().__init__()
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.MaxPool2D(2, 2),
      tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv2D(512, (1,1), padding="same", activation="relu"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation = "relu"),
      tf.keras.layers.Dense(20 * 20 * n_anchor_boxes * n_out, activation="sigmoid"),
      tf.keras.layers.Reshape((20, 20, n_anchor_boxes, n_out))
    ])
    self.model.summary()

  def call(self, x):
    return self.model(x)

class SimpleYolo(tf.keras.Model):
  def __init__(self, input_shape, n_out, n_class) -> None:
    super().__init__()
    self.n_class = n_class
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (7, 7), padding="same", input_shape=input_shape),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(192, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(128, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),
      
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), strides=(2,2), padding="valid"),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Dense(20 * 20 * n_out),
      tf.keras.layers.Reshape((20, 20, n_out))
    ])
    self.model.summary()

  def call(self, x):
    out = self.model(x)
    out_classes = tf.nn.sigmoid(out[..., :self.n_class])
    # out_coords = tf.keras.layers.ReLU()(out[..., self.n_class: self.n_class+4])
    out_coords = out[..., self.n_class: self.n_class+4]
    out_probs = tf.nn.sigmoid(out[..., self.n_class+4:])
    ret_val = tf.concat([out_classes, out_coords, out_probs], axis=-1)
    return ret_val
    # return out

def testSimpleYolo(input_shape=(320, 320, 3), n_anchor_boxes=9, n_out=12):
  model = SimpleYolo(input_shape, n_anchor_boxes, n_out, 7)
  x = tf.random.normal([16, 320, 320, 3])
  y_hat = model(x, training=False)
  print(y_hat.shape)

class SimpleYolo2(tf.keras.Model):
  def __init__(self, input_shape, n_out, n_class) -> None:
    super().__init__()
    self.n_class = n_class
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (7, 7), padding="same", input_shape=input_shape),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(192, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(128, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),
      
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), strides=(2,2), padding="valid"),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Dense(20 * 20 * n_out),
      tf.keras.layers.Reshape((20, 20, n_out))
    ])
    self.model.summary()

  def call(self, x):
    out = self.model(x)
    return out




class ConvBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding="same", alpha=0.1, **kwargs) -> None:
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.alpha = alpha
    self.padding=padding
  
  def build(self, input_shape):
    self.conv = layers.Conv2D(self.filters, self.kernel_size, padding=self.padding)

  def call(self, inputs):
    tensor = self.conv(inputs)
    tensor = layers.BatchNormalization()(tensor)
    out = layers.LeakyReLU(alpha=self.alpha)(tensor)
    return out

class YoloHead(tf.keras.layers.Layer):
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
  
  def build(self, input_shape):
    pass

  def call(self, inputs):
    tensor_class = inputs[..., :-5]
    # tensor_class = layers.Softmax()(tensor_class) # We dont use this because we
    # directly use tf.nn.softmax_cross_entropy_with_logits

    tensor_xy = inputs[..., -5:-3]
    tensor_xy = tf.nn.sigmoid(tensor_xy)

    tensor_wh = tf.exp(inputs[..., -3:-1])
    tensor_conf = inputs[..., -1:]
    tensor_conf = tf.nn.sigmoid(tensor_conf)

    tensor = tf.concat([tensor_class, tensor_xy, tensor_wh, tensor_conf], axis=-1)
    return tensor

def SimpleYolo3(input_shape, n_out, n_class):

  model = tf.keras.Sequential()
  model.add(L.Input(shape=input_shape))

  configs = [
    ["block", 32, (3,3)],
    ["block", 64, (3,3)],
    ["mp"],
    ["block", 64, (3,3)],
    ["block", 128, (3,3)],
    ["mp"],
    ["block", 128, (3,3)],
    ["block", 256, (3,3)],
    ["mp"],
    ["block", 256, (3,3)],
    ["block", 512, (3,3)],
    ["mp"],
    ["block", 512, (3,3)],
    ["block", 1024, (3,3)],
    ["conv", n_out, (3, 3)]
  ]

  for i, config in enumerate(configs):
    if config[0] == "mp":
      model.add(L.MaxPool2D(2,2, name=f"max_pool_{i}"))
    elif config[0] == "block":
      model.add(ConvBlock(config[1], config[2], alpha=0.1))
    elif config[0] == "conv":
      model.add(L.Conv2D(config[1], config[2], padding="same"))

  model.add(YoloHead())

  return model


def SimpleYolo4(input_shape, n_out, n_class):

  model = tf.keras.Sequential()
  model.add(L.Input(shape=input_shape))

  configs = [
    ["block", 32, (3,3)],
    ["block", 64, (3,3)],
    ["mp"],
    ["block", 64, (3,3)],
    ["block", 128, (3,3)],
    ["mp"],
    ["block", 128, (3,3)],
    ["block", 256, (3,3)],
    ["mp"],
    ["block", 256, (3,3)],
    ["block", 512, (3,3)],
    ["mp"],
    ["block", 512, (3,3)],
    ["block", 1024, (3,3)],
    ["conv", n_out, (1, 1)]
  ]

  for i, config in enumerate(configs):
    if config[0] == "mp":
      model.add(L.MaxPool2D(2,2, name=f"max_pool_{i}"))
    elif config[0] == "block":
      model.add(ConvBlock(config[1], config[2], padding="valid", alpha=0.1))
    elif config[0] == "conv":
      model.add(L.Conv2D(config[1], config[2], padding="valid"))

  model.add(YoloHead())

  return model


########### EXAMPLLE ##############
# https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

def _conv_block(inp, convs, skip=True):
  x = inp
  count = 0
  
  for conv in convs:
    if count == (len(convs) - 2) and skip:
      skip_connection = x
    count += 1
    
    if conv['stride'] > 1: x = layers.ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
    x = layers.Conv2D(conv['filter'], 
                conv['kernel'], 
                strides=conv['stride'], 
                padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                name='conv_' + str(conv['layer_idx']), 
                use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']: x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['leaky']: x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

  return layers.add([skip_connection, x]) if skip else x

def make_yolov3_model(n_class: int=81):
    input_image = layers.Input(shape=(*example_image_size, 3))

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  n_class+5, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': n_class+5, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 105}], skip=False)

    yolo_106 = layers.MaxPool2D(2,2)(yolo_106)
    yolo_106 = layers.Conv2D(n_class + 5, (3,3), padding="same")(yolo_106)

    out = YoloHead()(yolo_106)


    # model = tf.keras.Model(input_image, [yolo_82, yolo_94, yolo_106])  
    model = tf.keras.Model(input_image, out)
    return model
