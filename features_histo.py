from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd



n_features = 768

features = []
for i in range(n_features):
    features.append(i)

features.append('label')

df = pd.DataFrame(columns=features)

# USED FULL HISTOGRAM _____________________________________________________________________________________________________________
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
df.to_csv('dataset_histo.csv')
# _________________________________________________________________________________________________________________________________