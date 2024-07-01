# TENSOR FLOW 
#creating constant variable
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("TF_ENABLE_ONEDNN_OPTS"))


d = tf.constant(True)
print(d)

array = tf.constant(np.array([1,2,3,4]))
print(array)

t_2d = tf.constant([1,2,3,4], shape=(2,2,), dtype='int32')
print(t_2d)

t_3d = tf.constant([[[1,2],[2,3],[3,4] ]], dtype='float32')
print(t_3d)
print(type(t_3d))
print(t_3d.shape)
print(t_3d.numpy())

