import numpy as np


BATCH_SIZE = 8

image_size = (320, 320)
n_channels = 3
learning_rate = 1e-3
LAMBDA_COORD = 1.0
LAMBDA_NOOBJ = 0.5
LAMBDA_OBJ = 50.0
LAMBDA_CLASS = 1.0

n_cell_y = 20
n_cell_x = 20
cell_width = 1.0 / n_cell_x
cell_height = 1.0 / n_cell_y 
anchor_size = [32, 64, 128]
anchor_ratio = [(1, 1), (1, np.sqrt(2)), (np.sqrt(2), 1)]
n_anchor = len(anchor_size) * len(anchor_ratio)

CONFIDENCE_THHRESHOLD = 0.6

EPSILON = 1e-8