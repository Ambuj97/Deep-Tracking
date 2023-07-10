## Purpose (WHY):
## This python file handles the preprocessing of the video data which is originally
## in a .seq format and the annotations that are in .vbb format.

## Method (HOW):
## The .seq files are first converted into .png files (one for each frame in that video)
## which are later combined to create a .mp4 video file. The annotations .vbb files (which
## contain data for annotations, labels, pedestrian IDs, pedestrian positions) are
## converted to .txt files which contains data in a proper format.

## Outcome (WHAT):
## The end product after preprocessing is annotated video data which is suitable for
## ingestion into the YOLO model for pedestrian detection.

## Reference:
## This dataset has been maintained by Caltech Data Library
## Dollar, P., Wojek, C., Schiele, B., & Perona, P. (2009). Caltech Pedestrians [Data set].
## IEEE Conference on Computer Vision and Pattern Recognition.
## https://doi.org/10.1109/CVPR.2009.5206631


# importing necessary libraries
import os
import time
import glob

import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from tqdm import tqdm
tqdm.pandas()


# checking if the directories exist or not
if os.path.exists("Data/dataset") == False:
    os.mkdir("Data/dataset")

if os.path.exists("Data/dataset/images") == False:
    os.mkdir("Data/dataset/images")

if os.path.exists("Data/dataset/images/train") == False:
    os.mkdir("Data/dataset/images/train")

if os.path.exists("Data/dataset/images/validation") == False:
    os.mkdir("Data/dataset/images/validation")

if os.path.exists("Data/dataset/images/test") == False:
    os.mkdir("Data/dataset/images/test")

if os.path.exists("Data/dataset/labels") == False:
    os.mkdir("Data/dataset/labels")

if os.path.exists("Data/dataset/labels/train") == False:
    os.mkdir("Data/dataset/labels/train")

if os.path.exists("Data/dataset/labels/validation") == False:
    os.mkdir("Data/dataset/labels/validation")

if os.path.exists("Data/dataset/labels/test") == False:
    os.mkdir("Data/dataset/labels/test")

# the orignal work for proper conversion of data has been referenced from
# https://github.com/simonzachau/caltech-pedestrian-dataset-to-yolo-format-converter

# extracting annotations data from .vbb annotation files
def yoloBBFormatConv(imageDimensions, boxMeasurements):
    # the original data has annotations for top left bounding box x and y values along
    # with the height and width of the bounding box
    (topLeftX, topLeftY, boxWidth, boxHeight) = boxMeasurements

    # dimension of the image or the frame in the video
    imageWidth, imageHeight = imageDimensions




    

