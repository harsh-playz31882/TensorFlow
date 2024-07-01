import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("TF_ENABLE_ONEDNN_OPTS"))

a=tf.Variable(1)
print(a)

print(a.name)
print(a.shape)
print(a.numpyw())

x = tf.Variable([1.2, 3.14, 2.4])
print(x) 

t_f = tf.Variable([3+4j])
print(t_f)

t_con = tf.constant([1,2,3,4])
t_v = tf.Variable(t_con)
print(t_v) 

t_2d = tf.Variable([[2,2],[4,4]])
print(t_2d)

m = tf.argmax(t_2d)   #index of highest value
print(m)


tf.convert_to_tensor(t_2d)

t_2d.assign([[5,6],[7,8]])  #replaces with old memory location
print(t_2d)

t_2d.assign_add([[5,0],[1,1]])  #creates new memory location 
print(t_2d)                   