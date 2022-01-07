from types import new_class
import tensorflow as tf


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
  def __init__(self, filters, kernel_size, alpha=0.1, **kwargs) -> None:
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.alpha = alpha
  
  def build(self, input_shape):
    self.conv = L.Conv2D(self.filters, self.kernel_size, padding="same")

  def call(self, inputs):
    tensor = self.conv(inputs)
    tensor = L.BatchNormalization()(tensor)
    out = L.LeakyReLU(alpha=self.alpha)(tensor)
    return out

class YoloHead(tf.keras.layers.Layer):
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
  
  def build(self, input_shape):
    pass

  def call(self, inputs):
    tensor_class = inputs[..., :-5]
    # tensor_class = L.Softmax()(tensor_class) # We dont use this because we
    # directly use tf.nn.softmax_cross_entropy_with_logits

    tensor_xy = inputs[..., -5:-3]
    tensor_xy = tf.nn.sigmoid(tensor_xy)

    tensor_wh = inputs[..., -3:-1]
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