import numpy as np


BATCH_SIZE = 4

image_size = (400, 400)

example_image_size = (416, 416)

n_channels = 3
learning_rate = 1e-4
LAMBDA_COORD = 10.0
LAMBDA_NOOBJ = 0.1
LAMBDA_OBJ = 50.0
LAMBDA_CLASS = 1.0
LAMBDA_WH = 1.0

n_cell_y = 26
n_cell_x = 26
cell_width = 1.0 / n_cell_x
cell_height = 1.0 / n_cell_y 
anchor_size = [32, 64, 128]
anchor_ratio = [(1, 1), (1, np.sqrt(2)), (np.sqrt(2), 1)]
n_anchor = len(anchor_size) * len(anchor_ratio)

CONFIDENCE_THHRESHOLD = 0.6

EPSILON = 1e-8