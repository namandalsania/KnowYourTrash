import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Conv2D, Activation, GlobalMaxPooling2D, Concatenate



# Define the path to your Keras .h5 model
keras_model_path = 'C:/Users/Manan/Desktop/finaleff/Checkpoints/model-20.h5'

# Define the path and name for your TensorFlow Lite .tflite model
tflite_model_path = 'C:/Users/Manan/Desktop/finaleff/final.tflite'


# Load the Keras .h5 model
keras_model = tf.keras.models.load_model(keras_model_path)

# Convert the Keras .h5 model to TensorFlow Lite .tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the converted TensorFlow Lite .tflite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
