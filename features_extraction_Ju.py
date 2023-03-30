
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
import skimage
from skimage.color import rgb2gray
from skimage.morphology import binary_closing, binary_opening, disk

from scipy.stats import entropy, moment, mode

import matplotlib.pyplot as plt
import numpy as np


L_histo_median = []
L_histo_range = []
L_histo_variance = []
L_histo_deviation = []
L_histo_covariance = []
L_histo_moment = []
L_histo_entropy = []
L_sauvola_threshold = []
L_sauvola_ratio = []
L_seg_1 = []
L_seg_2 = []
L_seg_3 = []
L_seg_4 = []
L_seg_5 = []
L_seg_6 = []
L_seg_7 = []
L_seg_8 = []
L_seg_9 = []
L_seg_10 = []
L_seg_11 = []
L_seg_12 = []
L_seg_13 = []
L_seg_14 = []
L_label = []


features_dict = {
  "histo_median": L_histo_median,
  "histo_range": L_histo_range,
  "histo_variance": L_histo_variance,
  "histo_deviation": L_histo_deviation,
  "histo_covariance": L_histo_covariance,
  "histo_moment": L_histo_moment,
  "histo_entropy": L_histo_entropy,
  "sauvola_threshold": L_sauvola_threshold,
  "sauvola_ratio": L_sauvola_ratio,
  "seg_1": L_seg_1,
  "seg_2": L_seg_2,
  "seg_3": L_seg_3,
  "seg_4": L_seg_4,
  "seg_5": L_seg_5,
  "seg_6": L_seg_6,
  "seg_7": L_seg_7,
  "seg_8": L_seg_8,
  "seg_9": L_seg_9,
  "seg_10": L_seg_10,
  "seg_11": L_seg_11,
  "seg_12": L_seg_12,
  "seg_13": L_seg_13,
  "seg_14": L_seg_14,
  "label": L_label}


# USED FULL HISTOGRAM _____________________________________________________________________________________________________________
for i in range(2):
    # Extract histogram for healthy potatos
    if i == 0:
        files = [f for f in listdir('potato_dataset/train/healthy') if isfile(join('potato_dataset/train/healthy', f))]
        label = 0
    else:
        files = [f for f in listdir('potato_dataset/train/blight') if isfile(join('potato_dataset/train/blight', f))]
        label = 1

    for f in files:
        if i == 0:
            img = skimage.io.imread('potato_dataset/train/healthy/' + f)
        else:
            img = skimage.io.imread('potato_dataset/train/blight/' + f)
        
        histo = skimage.exposure.histogram(img)[0]

        # plt.imshow(img, cmap = "gray")
        # plt.show()
        
        gray_img = skimage.color.rgb2gray(img)
        sauvola_threshold = skimage.filters.threshold_sauvola(gray_img)
        sauvola_binarized = (gray_img > sauvola_threshold)*1
        sauvola_binarized_list = [item for sublist in sauvola_binarized for item in sublist]
        sauvola_ratio = sauvola_binarized_list.count(0) / sauvola_binarized_list.count(1)

        # Histogram features
        L_histo_median.append(np.median(histo))
        L_histo_range.append(np.max(histo - np.min(histo)))
        L_histo_variance.append(np.var(histo))
        L_histo_deviation.append(np.std(histo))
        L_histo_covariance.append(np.cov(histo))
        L_histo_moment.append(moment(histo))
        L_histo_entropy.append(entropy(histo))
        L_sauvola_threshold.append(np.mean(sauvola_threshold[0]))
        L_sauvola_ratio.append(sauvola_ratio)

        L_label.append(label)


# Segmentation features
df_seg = pd.read_csv('seg_features.csv')
for sample in range(len(df_seg.index)):
    features = df_seg.loc[sample, :].values.tolist()
    L_seg_1.append(features[0])
    L_seg_2.append(features[1])
    L_seg_3.append(features[2])
    L_seg_4.append(features[3])
    L_seg_5.append(features[4])
    L_seg_6.append(features[5])
    L_seg_7.append(features[6])
    L_seg_8.append(features[7])
    L_seg_9.append(features[8])
    L_seg_10.append(features[9])
    L_seg_11.append(features[10])
    L_seg_12.append(features[11])
    L_seg_13.append(features[12])
    L_seg_14.append(features[13])


df = pd.DataFrame(features_dict)
print(df)
df.to_csv("dataset.csv")
# _________________________________________________________________________________________________________________________________


"""
L_histo_mode = []
L_histo_mean = []
L_histo_median = []
L_histo_range = []
L_histo_variance = []
L_histo_deviation = []
L_histo_covariance = []
L_histo_moment = []
L_histo_entropy = []


features_dict = {
  "histo_mode": L_histo_mode,
  "histo_mean": L_histo_mean,
  "histo_median": L_histo_median,
  "histo_range": L_histo_range,
  "histo_variance": L_histo_variance,
  "histo_deviation": L_histo_deviation,
  "histo_covariance": L_histo_covariance,
  "histo_moment": L_histo_moment,
  "histo_entropy": L_histo_entropy}




# Open image
img_healthy = skimage.io.imread('potato_dataset/train/healthy/7bfda067-6e35-4af5-a9c4-4b3b5f357871___RS_HL 1813.JPG') 
img_disease = skimage.io.imread('potato_dataset/train/blight/c89d5a2a-2c1e-496e-84b8-b7a844a97ac9___RS_LB 4920.JPG') 


# HISTOGRAM FEATURES _________________________________________________________________________
histo = skimage.exposure.histogram(img_healthy)[0]

L_histo_mode.append(mode(histo))
L_histo_mean.append(np.mean(histo))
L_histo_median.append(np.median(histo))
L_histo_range.append(np.max(histo - np.min(histo)))
L_histo_variance.append(np.var(histo))
L_histo_deviation.append(np.std(histo))
L_histo_covariance.append(np.cov(histo))
L_histo_moment.append(moment(histo))
L_histo_entropy.append(entropy(histo))

# ____________________________________________________________________________________________
"""


"""

# Open image
img_healthy = skimage.io.imread('potato_dataset/train/healthy/7bfda067-6e35-4af5-a9c4-4b3b5f357871___RS_HL 1813.JPG') 
img_disease = skimage.io.imread('potato_dataset/train/blight/c89d5a2a-2c1e-496e-84b8-b7a844a97ac9___RS_LB 4920.JPG') 

gray_img_healthy = skimage.color.rgb2gray(img_healthy)
gray_img_disease = skimage.color.rgb2gray(img_disease)
hsv_img_healthy = skimage.color.rgb2hsv(img_healthy)
hsv_img_disease = skimage.color.rgb2hsv(img_disease)



# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

plt.subplot(2,2,1)
plt.title("Sauvola Thresholding healthy")
plt.imshow(img_healthy, cmap = "gray")

plt.subplot(2,2,2)
plt.title("Sauvola Thresholding healthy")
plt.imshow(img_disease, cmap = "gray")

# plt.show()



# OTSU __________________________________________________________________
# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))
 
# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold_healthy = skimage.filters.threshold_otsu(gray_img_healthy)
threshold_disease = skimage.filters.threshold_otsu(gray_img_disease)
 
# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_img_healthy = (gray_img_healthy > threshold_healthy)*1
binarized_img_disease = (gray_img_disease > threshold_disease)*1

binarized_img_healthy_morpho = binary_opening(binarized_img_healthy, disk(1))
binarized_img_disease_morpho = binary_opening(binarized_img_disease, disk(1))

plt.subplot(2,2,1)
plt.title("Otsu Thresholding Healthy - Threshold: >"+str(threshold_healthy))
plt.imshow(binarized_img_healthy, cmap = "gray")

plt.subplot(2,2,2)
plt.title("Otsu Thresholding Disease - Threshold: >"+str(threshold_disease))
plt.imshow(binarized_img_disease, cmap = "gray")

plt.subplot(2,2,3)
plt.title("Otsu Thresholding Healthy - Threshold: >"+str(threshold_healthy))
plt.imshow(binarized_img_healthy_morpho, cmap = "gray")

plt.subplot(2,2,4)
plt.title("Otsu Thresholding Disease - Threshold: >"+str(threshold_disease))
plt.imshow(binarized_img_disease_morpho, cmap = "gray")

plt.show()
# _______________________________________________________________________


# SAUVOLA _______________________________________________________________
# Setting plot size to 15, 15
# plt.figure(figsize=(15, 15))
 
# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold_healthy = skimage.filters.threshold_sauvola(gray_img_healthy)
threshold_disease = skimage.filters.threshold_sauvola(gray_img_disease)
print("____ TEST ____")
print("\nthreshold healthy: ____", np.mean(threshold_healthy[0]))
print("\nthreshold disease: ____", np.mean(threshold_disease[0]))
"""
"""
plt.subplot(2,2,1)
plt.title("Sauvola Thresholding healthy")
plt.imshow(threshold_healthy, cmap = "gray")

plt.subplot(2,2,2)
plt.title("Sauvola Thresholding disease")
plt.imshow(threshold_disease, cmap = "gray")

 
# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_img_healthy = (gray_img_healthy > threshold_healthy)*1
binarized_img_disease = (gray_img_disease > threshold_healthy)*1


plt.subplot(2,2,3)
plt.title("Sauvola Thresholding Healthy - Converting to 0's and 1's")
plt.imshow(binarized_img_healthy, cmap = "gray")

plt.subplot(2,2,4)
plt.title("Sauvola Thresholding Disease - Converting to 0's and 1's")
plt.imshow(binarized_img_disease, cmap = "gray")

plt.show()

values_healthy = [item for sublist in binarized_img_healthy for item in sublist]
ratio = values_healthy.count(0) / values_healthy.count(1)
print(ratio)

# _______________________________________________________________________

"""