# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:56:19 2019

@author: RUDRAJIT
"""

# Convolutional Neural Network

# Installing Keras
# conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3),activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
'''
    Performance can be improved either by adding more conv layers
    with more feature maps (>32,i.e. 64,128,...)
    OR
    Changing the Dim of channel in input_shape
    Recommended in GPU
    #1st layer
    classifier.add(Conv2D(32, (3,3), input_shape = (256,256,3),activation = 'relu'))
    #2nd layer
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    change target_size in preprocessing part to (256,256)
'''
    
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

from keras.models import load_model
classifier.save('model_file.h5')

'''
How to make a single prediction whether a speciﬁc picture contains a cat or a dog?
Inside the dataset folder you need to create a separate additional folder (let’s call it "single_prediction")
 containing the image (let’s call it "cat_or_dog.jpg") you want to predict and run the following code:

import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
test_image = image_utils.load_img(’dataset/single_prediction/cat_or_dog.jpg’,
                                  target_size = (64, 64))
test_image = image_utils.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis = 0) 
result = classifier.predict_on_batch(test_image) 
training_set.class_indices 
if result[0][0] == 1:
    prediction = ’dog’ 
else:
    prediction = ’cat’
'''