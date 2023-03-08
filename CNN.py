
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from imblearn.under_sampling import RandomUnderSampler

import keras.optimizers
from keras.saving.legacy.model_config import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras import models, layers, utils, backend as K
from keras.preprocessing.image import ImageDataGenerator



train_healthy_dir = os.path.join('potato_dataset/train/healthy')
train_blight_dir = os.path.join('potato_dataset/train/blight')
val_healthy_dir = os.path.join('potato_dataset/val/healthy')
val_blight_dir = os.path.join('potato_dataset/val/blight')

train_healthy_names = os.listdir(train_healthy_dir)
train_blight_names = os.listdir(train_blight_dir)
val_healthy_hames = os.listdir(val_healthy_dir)
val_blight_names = os.listdir(val_blight_dir)

print('total training healthy images:', len(os.listdir(train_healthy_dir)))
print('total training blight images:', len(os.listdir(train_blight_dir)))
print('total validation healthy images:', len(os.listdir(val_healthy_dir)))
print('total validation blight images:', len(os.listdir(val_blight_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_healthy_pic = [os.path.join(train_healthy_dir, fname) 
                for fname in train_healthy_names[pic_index-8:pic_index]]
next_blight_pic = [os.path.join(train_blight_dir, fname) 
                for fname in train_blight_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_healthy_pic + next_blight_pic):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)
plt.show()



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'potato_dataset/train',  # This is the source directory for training images
        classes = ['healthy', 'blight'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'potato_dataset/val',  # This is the source directory for training images
        classes = ['healthy', 'blight'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=19,
        # Use binary labels
        class_mode='binary',
        shuffle=False)


model = models.Sequential([layers.Flatten(input_shape = (200,200,3)), 
                                    layers.Dense(128, activation='relu'), 
                                    layers.Dense(1, activation='sigmoid')])

model.summary()

optimizer = keras.optimizers.Adam()
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      epochs=30,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
                      verbose=1)

print(preds)

"""
model = models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The third convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The fourth convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # # The fifth convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
    layers.Dense(1, activation='sigmoid')])

model.summary()

optimizer = keras.optimizers.Adam()
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
"""