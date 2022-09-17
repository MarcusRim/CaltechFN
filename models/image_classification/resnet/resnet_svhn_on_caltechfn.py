import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import scipy.io
import os

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
pretrained_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

history = model.fit(x=train_tensor, y=train_y_categorical, batch_size=64,epochs=40)

# test data processing
test_images = []
for file in os.listdir('images'): 
    img = Image.open('images/' + file)
    img = img.resize((32,32))
    img_arr = np.asarray(img)
    test_images.append(img_arr)

mat_test = scipy.io.loadmat('test.mat') # replace with path file to the .mat file 
test_labels = mat_test['y'][0]
del mat_test

test_tensor = tf.convert_to_tensor(test_images)
        
test_y_categorical = tf.keras.utils.to_categorical(test_labels, 10)

results = model.evaluate(test_tensor, test_y_categorical, batch_size=128)
print("test loss, test acc:", results)