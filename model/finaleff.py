import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define paths to the dataset
train_path = "C:/Users/Manan/Desktop/finaleff/waste-classification-data/DATASET/TRAIN"
valid_path = "C:/Users/Manan/Desktop/finaleff/waste-classification-data/DATASET/TEST"

# Define hyperparameters
dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

# Create generators for the training and validation datasets


# Set up data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=8, class_mode='categorical')
test_set = val_datagen.flow_from_directory(valid_path, target_size=(224, 224), batch_size=8, class_mode='categorical')

from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Add, Activation, Lambda, Concatenate, Multiply, BatchNormalization, Dropout, MaxPooling2D
from keras.models import Model
from keras import backend as K

def CBAM(inputs, reduction_ratio=8):
    # Channel attention module
    x = inputs
    channels = x.shape[-1]

    x_avg_pool = GlobalAveragePooling2D()(x)
    x_avg_pool = Reshape((1, 1, channels))(x_avg_pool)

    x_max_pool = GlobalMaxPooling2D()(x)
    x_max_pool = Reshape((1, 1, channels))(x_max_pool)

    x_avg_fc = Dense(channels // reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_avg_pool)
    x_avg_fc = Dense(channels, activation='linear', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_avg_fc)

    x_max_fc = Dense(channels // reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_max_pool)
    x_max_fc = Dense(channels, activation='linear', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_max_fc)

    x = Add()([x_avg_fc, x_max_fc])
    x = Activation('sigmoid')(x)

    # Spatial attention module
    z = inputs
    z_avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(z)
    z_max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(z)
    z = Concatenate(axis=3)([z_avg_pool, z_max_pool])

    z = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(z)

    x = Multiply()([x, z])

    return x


# Define the model architecture
input_shape = (224, 224, 3)
num_classes = 2

inputs = Input(shape=input_shape)

x = Conv2D(128, (3, 3), input_shape=input_shape)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = CBAM(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = CBAM(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = CBAM(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)


# Set up checkpoint to save the best model during training
MODEL_DIR = "C:/Users/Manan/Desktop/finaleff/Checkpoints"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))

# Set up early stopping to prevent overfitting
#earlystop = EarlyStopping(monitor='val_loss', patience=3)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[checkpoint]
)

# Save the final model
model.save("C:/Users/Manan/Desktop/finaleff/final.h5")

# Plot the loss and accuracy for training and validation
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(range(len(train_acc)), train_acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()