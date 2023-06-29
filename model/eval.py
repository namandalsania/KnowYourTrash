import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as k
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Conv2D, Activation, GlobalMaxPooling2D, Concatenate

# class CBAM(Layer):
#     def __init__(self, ratio=8, trainable=True, name=None, dtype=None):
#         super(CBAM, self).__init__(name=name, trainable=trainable)
#         self.ratio = ratio

#     def build(self, input_shape):
#         self.channel = input_shape[-1]
#         self.maxpool = GlobalMaxPooling2D()
#         self.avgpool = GlobalAveragePooling2D()

#         self.dense1 = Dense(units=self.channel//self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
#         self.dense2 = Dense(units=self.channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)
#         self.conv = Conv2D(filters=1, kernel_size=(7,7), strides=(1,1), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=True)

#         super(CBAM, self).build(input_shape)

#     def call(self, inputs):
#         maxpool = self.maxpool(inputs)
#         avgpool = self.avgpool(inputs)

#         x = Concatenate()([maxpool, avgpool])
#         x = self.dense1(x)
#         x = self.dense2(x)

#         attn = Reshape((1,1,self.channel))(x)
#         attn = Multiply()([inputs, attn])
#         attn = self.conv(attn)

#         return attn * inputs

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'ratio': self.ratio
#         })
#         return config
# custom_objects = {'CBAM': CBAM}


valid_path = "D:/KYTMP/waste-classification-data/DATASET/TEST"

test_datagen = ImageDataGenerator()

test_set = test_datagen.flow_from_directory(valid_path, target_size=(224, 224), batch_size=20, class_mode='categorical')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

model = tf.keras.models.load_model("C:/Users/Manan/Desktop/finaleff/final.h5")

loss, accuracy = model.evaluate(test_set)

print('Test loss:', loss)
print('Test accuracy:', accuracy)