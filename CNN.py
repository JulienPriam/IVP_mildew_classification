
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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# SCRIPT PARAMETERS ____________________________________________________________________________________________________
run_CNN = False
save_model = False
load_model = True

# hyperparameters tuning
n_features = 10
layer1_neurons = 15 # best 30
layer2_neurons = 12 # best 25
batch_size = 32 # best 128
epochs = 50
learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
# ______________________________________________________________________________________________________________________


# DATA PREPARATION _____________________________________________________________________________________________________
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
# ______________________________________________________________________________________________________________________


# RUN THE CNN __________________________________________________________________________________________________________
if run_CNN:
      model = Sequential()
      model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))

      model.add(Conv2D(32, (3, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))

      model.add(Conv2D(64, (3, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))

      model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
      model.add(Dense(64))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      model.add(Dense(1))
      model.add(Activation('sigmoid'))

      model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

      training = model.fit(train_generator, epochs=epochs, validation_data = validation_generator)


      # PLOT THE TRAINING HISTORY 
      metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
      fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

      # Training
      ax[0].set(title="Training")
      ax[0].set_ylim(0, 2)
      ax11 = ax[0].twinx()
      ax[0].plot(training.history['loss'], color='black')
      ax[0].set_xlabel('Epochs')
      ax[0].set_ylabel('Loss', color='black')
      for metric in metrics:
            ax11.plot(training.history[metric], label=metric)
            ax11.set_ylabel("Score", color='steelblue')
      ax11.legend()

      # Validation
      ax[1].set(title="Validation")
      ax22 = ax[1].twinx()
      ax[1].plot(training.history['val_loss'], color='black')
      ax[1].set_xlabel('Epochs')
      ax[1].set_ylabel('Loss', color='black')
      for metric in metrics:
            ax22.plot(training.history['val_' + metric], label=metric)
            ax22.set_ylabel("Score", color="steelblue")
      plt.show()
# ______________________________________________________________________________________________________________________


# SAVE THE MODEL _______________________________________________________________________________________________________
if save_model & run_CNN:
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_CNN.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_CNN.h5")
    print("\nSaved model to disk")     
# ______________________________________________________________________________________________________________________

# LOAD MODEL FROM DISK AND EVALUATE ON TESTING SET _____________________________________________________________________
if load_model:
    # load json and create model
    json_file = open('model_CNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_CNN.h5")
    print("\nLoaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    score = loaded_model.evaluate(validation_generator, verbose=0)
    print("{}: {}%".format(loaded_model.metrics_names[1], score[1] * 100))
# ______________________________________________________________________________________________________________________
