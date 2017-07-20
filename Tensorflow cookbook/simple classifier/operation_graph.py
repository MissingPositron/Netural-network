# operations on a computational graph
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m = tf.constant(3.)

# Multiplication
prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict = {x_data: x_vals}))

merged = tf.summary.merge_all()
my_writer = tf.summary.FileWriter('2_tensorflow_way.jpg',sess.graph)
