import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import scipy.io

# train data processing
mat_train = scipy.io.loadmat('train_32x32.mat') # replace with path file to the .mat file 
og_images = mat_train['X']
labels = mat_train['y']
images = []
for i in range(len(og_images[0][0][0])):
    images.append(og_images[:,:,:,i])
images = np.array(images)
labels = np.array(labels)
del mat_train
del og_images

train_tensor = tf.convert_to_tensor(images)

# need to convert the '10' label to '0'
for i in range(len(labels)):
    if labels[i] == 10:
        labels[i] = 0
        
train_y_categorical = tf.keras.utils.to_categorical(labels, 10)

model = Sequential()
pretrained_model = tf.keras.applications.densenet.DenseNet121(weights=None, include_top=False, input_shape=(32, 32, 3))
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.0001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

history = model.fit(x=train_tensor, y=train_y_categorical, batch_size=16, epochs=40)

# test data processing

mat_test = scipy.io.loadmat('test_32x32.mat') # replace with path file to the .mat file 
og_test_images = mat_test['X']
test_labels = mat_test['y']
test_images = []
for i in range(len(og_test_images[0][0][0])):
    test_images.append(og_test_images[:,:,:,i])
test_images = np.array(test_images)
test_labels = np.array(test_labels)
del mat_test
del og_test_images

test_tensor = tf.convert_to_tensor(test_images)

for i in range(len(test_labels)):
    if test_labels[i] == 10:
        test_labels[i] = 0
        
test_y_categorical = tf.keras.utils.to_categorical(test_labels, 10)

results = model.evaluate(test_tensor, test_y_categorical, batch_size=128)
print("test loss, test acc:", results)