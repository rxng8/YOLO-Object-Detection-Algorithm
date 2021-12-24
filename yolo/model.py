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
    out_classes = tf.keras.layers.Softmax()(out[..., :self.n_class])
    out_coords = tf.keras.layers.ReLU()(out[..., self.n_class: self.n_class+4])
    out_probs = tf.sigmoid(out[..., self.n_class+4:])
    ret_val = tf.concat([out_classes, out_coords, out_probs], axis=-1)
    return ret_val

def testSimpleYolo(input_shape=(320, 320, 3), n_anchor_boxes=9, n_out=12):
  model = SimpleYolo(input_shape, n_anchor_boxes, n_out, 7)
  x = tf.random.normal([16, 320, 320, 3])
  y_hat = model(x, training=False)
  print(y_hat.shape)