import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("TF_ENABLE_ONEDNN_OPTS"))

tf.compat.v1.disable_eager_execution()  #executing from tensor flow 1

t_1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(100,100))

t_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(100,100))

a = tf.add(t_1, t_2)
print(a)

ones_array = np.ones((100,100), np.float32)
print(ones_array)

with tf.compat.v1.Session() as sess:  #again executin Session command from tensor flow 1
    d = sess.run(a, feed_dict={t_1:ones_array, t_2:ones_array})
    print(d)

    



