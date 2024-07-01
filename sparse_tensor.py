import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("TF_ENABLE_ONEDNN_OPTS"))

''' Syntax
 tf.sparse.SparseTensor(indices, values, dense_shape)'''

st1 = tf.SparseTensor(indices=[[2,2],[4,5]], values=[10,20], dense_shape=[3,4])
print(st1)

#creating sparse tensor from dense
np_ar = np.array([[1,0,0,0],
                  [1,0,0,0],
                  [1,0,0,0],
                  [1,0,0,0],])
print(np_ar)
tf.sparse.from_dense(np_ar)

st2_fd = tf.sparse.from_dense(np_ar)  #dense tensor to sparse tensor
print(st2_fd)


print(st2_fd.values.numpy().tolist())
print(st2_fd.indices.numpy().tolist())
print(st2_fd.dense_shape.numpy().tolist())  

dt_fst = tf.sparse.to_dense(st2_fd) #sparse tensor to dense tensor
print(dt_fst)

print(dt_fst.numpy()) #dense tensor to numpy


st_add = tf.sparse.add(st2_fd, st2_fd)
print(st_add)
print(tf.sparse.to_dense(st_add).numpy())