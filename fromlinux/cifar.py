import tensorflow as tf
from tensorflow import keras

keras_file = './fromlinux/linux.h5'
#keras.models.save_model(model, keras_file)
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("linux.tflite", "wb").write(tflite_model)


