import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], True)
#import IPython
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
#from IPython import display
from collections import OrderedDict
import glob
import time
from tqdm import tqdm
import SimpleITK as sitk
import random


### variables

s1=24 #88 #18
s2=32 #72 #22
features_root=16
sample_shape = (24,32,1)
l_rate = 1e-4
BATCH_SIZE = 32
###


### Functions

def tra_pre(path, lst):
    tra = sorted(os.listdir(path))
    tr_in = list(np.zeros((len(lst), 18, 22, 1), dtype=np.uint16))
    for n, i in tqdm(enumerate(tra), total=len(tra)):
        im = sitk.ReadImage(os.path.join(path, i))
        t1 = sitk.GetArrayFromImage(im)
        tr_in[n] = t1
        n += 1
    return tr_in

def all(input):
    all = []
    for i in range(0, len(input)):
        patient = input[i]
        for img in range(0, len(patient)):
            all.append(patient[img])
    return all

def aug(inp):
  v=[]
  for i in range(0,len(inp)):
    im = inp[i]
    r_image = rotate(im, angle=-20 )
    rr_image = rotate(im, angle=20 )
    transform1 = AffineTransform(translation=(-3,3))
    transform2 = AffineTransform(translation=(3,-3))
    transform3 = AffineTransform(translation=(3,3))
    transform4 = AffineTransform(translation=(-3,-3))
    warp_image1 = warp(im,transform1)
    warp_image2 = warp(im,transform2)
    warp_image3 = warp(im,transform3)
    warp_image4 = warp(im,transform4)
    v.append(im)
    v.append(r_image)
    v.append(rr_image)
    v.append(warp_image1)
    v.append(warp_image2)
    v.append(warp_image3)
    v.append(warp_image4)
  return v

###

### Train dataset

path10_near = 'input_path'
path30_near = 'label_path'
train10_list = sorted(next(os.walk(path10_near))[2])
train30_list = sorted(next(os.walk(path30_near))[2])

input = tra_pre(path10_near, train10_list)
input = np.array(input)

output = tra_pre(path30_near, train30_list)
output = np.array(output)

all_input = all(input)
all_output = all(output)

new_in = aug(all_input)
new_out = aug(all_output)

# zip
zip_list = list (zip(new_in, new_out))
# shuffle
random.shuffle(zip_list)
# dataset
train_Images = zip_list

train_set, val_set = sklearn.model_selection.train_test_split(train_Images, train_size = 0.85)

train_d, label_d = zip (*train_set)
train_d = np.array(train_d)
label_d = np.array(label_d)
train_d = train_d.reshape(train_d.shape[0],24,32,1).astype('float32')
label_d = label_d.reshape(label_d.shape[0],24,32,1).astype('float32')

### Validation dataset

val_i, val_o = zip (*val_set)
val_i = np.array(val_i)
val_o = np.array(val_o)
val_i = val_i.reshape(val_i.shape[0],24,32,1).astype('float32')
val_o = val_o.reshape(val_o.shape[0],24,32,1).astype('float32')
###

# Batch and shuffle the data
train_data = tf.data.Dataset.from_tensor_slices(train_d).batch(BATCH_SIZE)
label_data = tf.data.Dataset.from_tensor_slices(label_d).batch(BATCH_SIZE)
train_dataset = list (zip(train_data,label_data))

## Test data

path10_resize_test= 'input_test_path'
path30_resize_test= 'label_test_path'

test10_list = sorted(next(os.walk(path10_resize_test))[2])
test30_list = sorted(next(os.walk(path30_resize_test))[2])

test = tra_pre(path10_resize_test, test10_list)
test = np.array(test)

label_test = tra_pre(path30_resize_test, test30_list)
label_test = np.array(label_test)
test_input = all(test)
test_output = all(label_test)

all_test_input = np.asarray(test_input)
all_test_input = all_test_input.reshape(all_test_input.shape[0],24,32,1).astype('float32')
all_test_output = np.asarray(test_output)
all_test_output = all_test_output.reshape(all_test_output.shape[0],24,32,1).astype('float32')
###

##Conv. Net

# Down _ Sample

#1
l1 = tf.keras.layers.Input(shape=( 24 , 32 ,1))
l2 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(3,3) , strides=(1, 1), padding='same')(l1) #, name='in_conv1_%d'%layer
l3 = tf.keras.layers.BatchNormalization( momentum=0.9, center=False, moving_mean_initializer='zero')(l2)
l4 = tf.keras.layers.LeakyReLU(alpha=0.3)(l3)

l5 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(3,3), strides=(1, 1), padding='same' )(l4)
l6 = tf.keras.layers.BatchNormalization( momentum=0.9, center=False, moving_mean_initializer='zero')(l5)
l7 = tf.keras.layers.LeakyReLU(alpha=0.3)(l6)

#2
l8 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(3,3) ,strides=(1, 1), padding='same')(l7) #, name='in_conv1_%d'%layer
l9 = tf.keras.layers.BatchNormalization( momentum=0.9, center=False, moving_mean_initializer='zero')(l8)
l10 = tf.keras.layers.LeakyReLU(alpha=0.3)(l9)

# up _ Sampling

#1
l61 = tf.keras.layers.Add()([l10, l7])
l38 = tf.keras.layers.Conv2D( 32 , kernel_size=3, strides=(1, 1), padding='same')(l61)
l39 = tf.keras.layers.BatchNormalization(momentum=0.9, center=False, moving_mean_initializer='zero')(l38)
l40 = tf.keras.layers.LeakyReLU(alpha=0.3)(l39)
l41 = tf.keras.layers.Add()([l40, l4])


#0

l49 = tf.keras.layers.Conv2D( 1 , kernel_size=3, strides=(1, 1), padding='same')(l41)
l50 = tf.keras.layers.BatchNormalization(momentum=0.9, center=False, moving_mean_initializer='zero')(l49)
l51 = tf.keras.layers.LeakyReLU(alpha=0.3)(l50)

output_map = tf.keras.layers.Add()([l51, l1])
###
generator = tf.keras.Model(inputs = l1 , outputs = output_map)
# generator.summary()
###

generator_optimizer = tf.keras.optimizers.Adam(learning_rate= l_rate)

### Training

generator.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.MeanSquaredError())  # {tf.keras.metrics.MeanSquaredError(),[PSNR],[SSIM]})
# loss='mean_squared_error',


history = generator.fit(train_d, label_d, epochs=2000, verbose=1,
                        validation_data=(val_i, val_o))
### Prediction
predictions = generator(all_test_input, training=False)

### Evaluation
test_loss, test_psnr,test_ssim,test_rmse = generator.evaluate(all_test_input,  all_test_output, verbose=2)

