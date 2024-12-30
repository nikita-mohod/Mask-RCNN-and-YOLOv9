import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Root directory of the project
#ROOT_DIR = os.path.abspath("../")
ROOT_DIR="E:/Nikita Mohod/Mask_RCNN/"

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
    #utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display();


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('E:/Nikita Mohod/Mask_RCNN/samples/mask_rcnn_coco.h5', by_name=True)

# COCO Class names
##class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
##               'bus', 'train', 'truck', 'boat', 'traffic light',
##               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
##               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
##               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
##               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
##               'kite', 'baseball bat', 'baseball glove', 'skateboard',
##               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
##               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
##               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
##               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
##               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
##               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
##               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
##               'teddy bear', 'hair drier', 'toothbrush']

class_names = ['BG', 'person', 'fire hydrant', 'bench', 
               'cat', 'dog', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'pizza',
               'chair', 'couch', 'potted plant', 'dining table', 'tv', 'laptop', 'remote',
               'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors',
               'teddy bear']


### Load a random image from the images folder
##image = skimage.io.imread('E:/Nikita Mohod/Mask_RCNN/images/2516944023_d00345997d_z.jpg')
##
###Original image
##plt.figure(figsize=(12,10))
##skimage.io.imshow(image)
##
### Run detection
##results = model.detect([image], verbose=1)
##
### Visualize results
##r = results[0]
##visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
##
##mask = r['masks']
##mask = mask.astype(int)
##mask.shape
##
##for i in range(mask.shape[2]):
##    temp = skimage.io.imread('E:/Nikita Mohod/Mask_RCNN/images/2516944023_d00345997d_z.jpg')
##    for j in range(temp.shape[2]):
##        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
##    plt.figure(figsize=(12,10))
##    plt.imshow(temp)

import cv2
 
### Opens the Video file
##cap= cv2.VideoCapture('E:/Nikita Mohod/11.avi')
##i=0
##while(cap.isOpened()):
##    ret, frame = cap.read()
##    if ret == False:
##        break
##    cv2.imwrite('kang'+str(i)+'.jpg',frame)
##    i+=1
## 
##cap.release()
##cv2.destroyAllWindows()

image = skimage.io.imread('C:/Users/Pratik/Documents/kang14750.jpg')

#Original image
plt.figure(figsize=(12,10))
skimage.io.imshow(image)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

mask = r['masks']
mask = mask.astype(int)
mask.shape

for i in range(mask.shape[2]):
    temp = skimage.io.imread('C:/Users/Pratik/Documents/kang14750.jpg')
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(12,10))
    plt.imshow(temp)
