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
  
  # Compute ious
  ious = dynamic_iou(true[..., -5:-1], pred[..., -5:-1]) # Shape (batch, 20,20)
  # ious = ious[..., tf.newaxis] # (batch, 20, 20, 1)
  label_conf = true[..., -1] * ious

  # Coordinates x, y loss
  # loss_xy shape (batch, 20, 20)
  loss_xy = LAMBDA_COORD * tf.reduce_sum(tf.square(true[..., -5:-3]*tf.expand_dims(identity_obj,-1) - pred[..., -5:-3]*tf.expand_dims(identity_obj,-1)), axis=-1)
  # loss_wh shape (batch, 20, 20)
  loss_wh = LAMBDA_COORD * tf.reduce_sum(tf.square(tf.sign(true[..., -3:-1])*tf.sqrt(tf.abs(true[..., -3:-1]) + 1e-6)*tf.expand_dims(identity_obj,-1) - tf.sign(pred[..., -3:-1])*tf.sqrt(tf.abs(pred[..., -3:-1]) + 1e-6)*tf.expand_dims(identity_obj,-1)), axis=-1) 
  # loss_class shape (batch, 20, 20)
  loss_class = tf.reduce_sum(tf.square(true[..., :-5] - pred[..., :-5]) * tf.expand_dims(identity_obj, -1), axis=-1)

  # loss_conf shape (batch, 20, 20)
  loss_conf = tf.square(label_conf * identity_obj - pred[..., -1] * identity_obj) \
    + LAMBDA_NOOBJ * tf.square(label_conf * (1 - identity_obj) - pred[..., -1] * (1 - identity_obj))

  # element wise addition
  loss = (loss_xy + loss_wh + loss_class + loss_conf)

  batch_loss = tf.reduce_sum(loss) / BATCH_SIZE
  return batch_loss



def yolo_loss_2(y_true, y_pred):
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
  coord_mask = y_true[..., -1:] * LAMBDA_COORD # Shape (batch, 20, 20, 1)

  # conf_mask
  conf_mask = tf.zeros(mask_shape)

  # class mask
  class_mask = y_true[..., -1] * LAMBDA_CLASS # Shape (batch, 20, 20)

  # Adjust the label confidence by multiplying the labeled confidence with the actual iou after predicted
  ious = dynamic_iou(y_true[..., -5:-1], y_pred[..., -5:-1]) # Shape (batch, 20, 20)
  true_box_conf = true_box_conf * ious # Shape (batch, 20, 20) x (batch, 20, 20) = (batch, 20, 20)

  conf_mask += tf.cast(ious < CONFIDENCE_THHRESHOLD, dtype=tf.float32) * (1 - y_true[..., -1]) * LAMBDA_NOOBJ

  conf_mask += y_true[..., -1] * LAMBDA_OBJ

  # Finalize the loss

  # compute the number of position that we are actually backpropagating
  nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
  nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, dtype=tf.float32))
  nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

  loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse
  loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + EPSILON) / 2. # divide by two cuz that's the mse
  loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + EPSILON) / 2.
  loss_class = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(true_box_class, pred_box_class, axis=-1) * class_mask
  ) / nb_class_box

  loss = loss_xy + loss_wh + loss_conf + loss_class
  return loss, [loss_xy, loss_wh, loss_conf, loss_class]