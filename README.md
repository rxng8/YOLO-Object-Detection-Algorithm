# YOLO Object Detection Algorithm Training and Evaluating Pipeline

## Requirement:
1. System: Window. Linux and mac not supported (This is because the kernel and libraries in `requirements.txt` are specialized to window)
2. Dataset:
 * Card dataset: [https://www.kaggle.com/luantm/playing-card](https://cocodataset.org/#download)
 * COCO dataset: [https://cocodataset.org/#download](https://cocodataset.org/#download)
3. VS Code (in order to run the jupyter notebook with python file: VS Code format)

## File structure:
* `yolo`: Contains the core library of the algorithm.
  * `__init__.py`
  * `const.py`: Storing hyper params.
  * `loss.py`: Collection of loss functions
  * `model.py`: Collection of YOLO model and base conv model.
  * `utils.py`: Utillities methods.
* `main_card_ds.py`: This is the jupyter notebook (vscode format) to train the YOLO v1 algorithm on the solitare card detection dataset.
* `main_coco_ds.py`: This is the jupyter notebook (vscode format) to train the YOLO v1 algorithm on the coco object detection dataset.

## Run the training process:

* First, install the required libraries: 
```
conda create -n tf
conda activate tf
pip install -r requirements.txt
```
* Next, go into one python file `main_card_ds.py` or `main_coco_ds.py` and experiment with the VS Code notebook.

## Reference:
```
@INPROCEEDINGS{7780460,
author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
title={You Only Look Once: Unified, Real-Time Object Detection}, 
year={2016},
volume={},
number={},
pages={779-788},
doi={10.1109/CVPR.2016.91}}
```