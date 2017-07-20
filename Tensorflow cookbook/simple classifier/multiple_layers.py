# Layering nested operations
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors
# Create a small random "image" of size 4x4
x_shape = [1, 4, 4, 1]
x_vals = np.random.uniform(size = x_shape)

x_data = tf.placeholder(tf.float32, shape = x_shape)

# Create a layer that takes a spatial moving window average
# Our window will be 2x2 with a stride of 2 for height and width
# The filter value will be 0.25 because we want the average of the 2x2 window
my_filter = tf.constant(0.25, shape = [2, 2, 1, 1])
my_stride = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_stride,
                            padding = 'SAME', name = 'Moving_Avg_Window')

# Define a custom layer which will be sigmoid(Ax+b) where 
# x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape = [2,2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    return (tf.sigmoid(temp))

# Add custom layer to graph
with tf.name_scope('Custom_layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

# The output should be an array that is 2x2, but size (1, 2, 2, 1)
print(sess.run(mov_avg_layer, feed_dict = {x_data: x_vals}))

# After custom operation, size is now 2x2
print(sess.run(custom_layer1, feed_dict = {x_data: x_vals}))

merged = tf.summary.merge_all()
my_writer = tf.summary.FileWriter('2_tensorflow_way', sess.graph)