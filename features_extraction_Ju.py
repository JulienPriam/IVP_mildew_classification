
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd

features = []
for i in range(768):
    features.append(i)

features.append('label')

df = pd.DataFrame(columns=features)

# Extract histogram for healthy potatos
files = [f for f in listdir('potato_dataset/Potato___healthy') if isfile(join('potato_dataset/Potato___healthy', f))]
for f in files:
    img = Image.open('potato_dataset/Potato___healthy/' + f, 'r')
    sample = img.histogram()
    sample.append(0)
    df.loc[len(df)] = sample

# Extract histogram for diseased potatos
files = [f for f in listdir('potato_dataset/Potato___Late_blight') if isfile(join('potato_dataset/Potato___Late_blight', f))]
for f in files:
    img = Image.open('potato_dataset/Potato___Late_blight/' + f, 'r')
    sample = img.histogram()
    sample.append(1)
    df.loc[len(df)] = sample

print(df)
df.to_csv('dataset.csv')