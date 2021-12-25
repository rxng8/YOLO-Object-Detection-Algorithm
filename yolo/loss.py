import tensorflow as tf
from .const import *
from .utils import iou, dynamic_iou

def simple_mse_loss(true_label, pred_label, n_class):
  # true label has the shape (batch_size, n_box_y, n_box_x, n_anchor, n_class + 5). (32, 20, 20, 9, 12)
  cond_object = true_label[:, :, :, :, n_class + 4] == 1 # Shape: (batch_size, n_box_y, n_box_x, n_anchor)
  cond_object = tf.expand_dims(cond_object, -1)
  cond_object = tf.broadcast_to(cond_object, (*cond_object.shape[:-1], n_class + 5))
  
  # We also compute for false label
  cond_object = cond_object.numpy()
  cond_object[:, :, :, :, n_class + 4] = True

  # Compute mean squared error for the plan-to-computed indices.
  mse = tf.reduce_mean(tf.square(true_label[cond_object] - pred_label[cond_object]))
  return mse

def yolo_loss(true, pred):
  # reference: https://blog.emmanuelcaradec.com/humble-yolo-implementation-in-keras/
  # (batch_size, n_box_y, n_box_x, n_anchor, n_out)
  n_anchor_boxes = true.shape[3]

  
  # Get the best box iou
  # for box in range(n_anchor_boxes):
  #   current_iou = iou(
  #     anchor_box_xmin,
  #     anchor_box_ymin,
  #     anchor_box_xmax,
  #     anchor_box_ymax,
  #     true[],
  #     resized_ymin,
  #     resized_xmax,
  #     resized_ymax
  #   )

  identity_obj = true[..., -1].numpy() # (batch, 20, 20)
  # Shape (batch, 20, 20, n_classes + 5)
  
  # Coordinates x, y loss
  # loss_xy shape (batch, 20, 20)
  loss_xy = LAMBDA_COORD * tf.reduce_sum(tf.square(true[..., -5:-3]*tf.expand_dims(identity_obj,-1) - pred[..., -5:-3]*tf.expand_dims(identity_obj,-1)), axis=-1)
  # loss_wh shape (batch, 20, 20)
  loss_wh = LAMBDA_COORD * tf.reduce_sum(tf.square(tf.sign(tf.sqrt(tf.abs(true[..., -3:-1]) + 1e-6))*tf.expand_dims(identity_obj,-1) - tf.sign(tf.sqrt(tf.abs(pred[..., -3:-1]) + 1e-6))*tf.expand_dims(identity_obj,-1)), axis=-1) 
  # loss_class shape (batch, 20, 20)
  loss_class = tf.reduce_sum(tf.square(true[..., :-5] - pred[..., :-5]) * tf.expand_dims(identity_obj, -1), axis=-1) 
  
  # loss_conf shape (batch, 20, 20)
  loss_conf = tf.square(true[..., -1] * identity_obj - pred[..., -1] * identity_obj) \
    + LAMBDA_NOOBJ * tf.square(true[..., -1] * (1 - identity_obj) - pred[..., -1] * (1 - identity_obj))

  # iou shape: (batch_size, n_cell_y, n_cell_x)
  # ious = dynamic_iou(true[..., -5:-1], pred[..., -5:-1])
  # loss_conf = tf.square(ious*true[..., -1] - pred[..., -1]) * true[..., -1] \
  #   + LAMBDA_NOOBJ * tf.square(ious*true[..., -1] - pred[..., -1]) * (1 - true[..., -1])

  # element wise addition
  loss = (loss_xy + loss_wh + loss_class + loss_conf)
  batch_loss = tf.reduce_sum(loss) / BATCH_SIZE
  return batch_loss