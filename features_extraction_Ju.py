
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd




# SCRIPT PARAMETERS _______________________________________________________________________________________________________________
reduce_histogram = True
# _________________________________________________________________________________________________________________________________

if reduce_histogram:
    n_features = 47
else:
    n_features = 768

features = []
for i in range(n_features):
    features.append(i)

features.append('label')

df = pd.DataFrame(columns=features)

# USED FULL HISTOGRAM _____________________________________________________________________________________________________________
if not reduce_histogram:
    # Extract histogram for healthy potatos
    files = [f for f in listdir('potato_dataset/train/healthy') if isfile(join('potato_dataset/train/healthy', f))]
    for f in files:
        img = Image.open('potato_dataset/train/healthy/' + f, 'r')
        sample = img.histogram()
        sample.append(0)
        df.loc[len(df)] = sample

    # Extract histogram for diseased potatos
    files = [f for f in listdir('potato_dataset/train/blight') if isfile(join('potato_dataset/train/blight', f))]
    for f in files:
        img = Image.open('potato_dataset/train/blight/' + f, 'r')
        sample = img.histogram()
        sample.append(1)
        df.loc[len(df)] = sample

    print(df)
    df.to_csv('dataset.csv')
# _________________________________________________________________________________________________________________________________


# USED REDUCED HISTOGRAM __________________________________________________________________________________________________________
if reduce_histogram:
    # Extract histogram for healthy potatos
    files = [f for f in listdir('potato_dataset/train/healthy') if isfile(join('potato_dataset/train/healthy', f))]
    for f in files:
        img = Image.open('potato_dataset/train/healthy/' + f, 'r')
        sample = img.histogram()
        reduced_sample = []
        sum = 0
        for i in range(len(sample)):
            if (i % 16 == 0) & (i != 0):
                reduced_sample.append(sum)
                sum = 0
            sum += sample[i]
        reduced_sample.append(0)
        df.loc[len(df)] = reduced_sample

    # Extract histogram for diseased potatos
    files = [f for f in listdir('potato_dataset/train/blight') if isfile(join('potato_dataset/train/blight', f))]
    for f in files:
        img = Image.open('potato_dataset/train/blight/' + f, 'r')
        sample = img.histogram()
        reduced_sample = []
        sum = 0
        for i in range(len(sample)):
            if (i % 16 == 0) & (i != 0):
                reduced_sample.append(sum)
                sum = 0
            sum += sample[i]
        reduced_sample.append(1)
        df.loc[len(df)] = reduced_sample

    print(df)
    df.to_csv('dataset.csv')
# _________________________________________________________________________________________________________________________________


"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from os import listdir
from os.path import isfile, join

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from skimage import io
from skimage.color import rgb2gray



features = []
for i in range(10):
    features.append(i)

features.append('label')
df = pd.DataFrame(columns=features)


# Extract histogram for healthy potatos
files = [f for f in listdir('potato_dataset/train/healthy') if isfile(join('potato_dataset/train/healthy', f))]
for f in files:
    img = io.imread('potato_dataset/train/healthy/' + f)
    img = rgb2gray(img)
    edge_img = filters.sobel(img)
    sample = np.histogram(edge_img.flatten())[0].tolist()
    sample.append(0)

    df.loc[len(df)] = sample

# Extract histogram for diseased potatos
files = [f for f in listdir('potato_dataset/train/blight') if isfile(join('potato_dataset/train/blight', f))]
for f in files:
    img = io.imread('potato_dataset/train/blight/' + f)
    img = rgb2gray(img)
    edge_img = filters.sobel(img)
    sample = np.histogram(edge_img.flatten())[0].tolist()
    sample.append(1)

    df.loc[len(df)] = sample


print(df)
df.to_csv('dataset_histo.csv')
print('\nDataset saved as dataset_histo.csv\n')


"""