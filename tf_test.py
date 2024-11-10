import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
print(tf.test.is_built_with_cuda())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())