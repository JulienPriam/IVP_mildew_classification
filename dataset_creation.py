import cv2
import numpy as np
import matplotlib.pyplot as plt

EARLY_PATH='potato_dataset/early_blight'
LATE_PATH='potato_dataset/late_blight'
HEALTHY_PATH='potato_dataset/healthy'

im=cv2.imread('/home/salome/IVP_mildew_classification/potato_dataset/early_blight/0a47f32c-1724-4c8d-bfe4-986cedd3587b___RS_Early.B 8001.JPG')