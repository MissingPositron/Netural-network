# Activation functions

# Implementation activation functions
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
#import tensorflow.nn as nn 
from tensorflow.python.framework import ops 
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# X range 
x_vals = np.linspace(start = -10., stop = 10., num = 100)

# ReLU activation
print(sess.run(tf.nn.relu([-3., 3., 10.])))
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation
print(sess.run(tf.nn.relu6([-3., 3., 10.])))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation
print(sess.run(tf.nn.tanh([-1., 0., 1.])))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation
print(sess.run(tf.nn.softsign([-1., 0., 1.])))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation
print(sess.run(tf.nn.softplus([-1., 0., 1.])))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation
print(sess.run(tf.nn.elu([-1.0, 0, 1.0])))
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.plot(x_vals, y_relu, 'r--', label = 'ReLU', linewidth = 2)
plt.plot(x_vals, y_relu6, 'b--', label = 'ReLU6', linewidth = 2)
plt.plot(x_vals, y_sigmoid, 'g--', label = 'sigmoid', linewidth = 2)
plt.plot(x_vals, y_tanh, 'k--', label = 'tanh', linewidth = 2)
plt.plot(x_vals, y_softsign, 'm--', label = 'softsign', linewidth = 2)
plt.plot(x_vals, y_softplus, 'y--', label = 'softplus', linewidth = 2)
plt.plot(x_vals, y_elu, 'r:', label = 'elu', linewidth = 2)
plt.legend(loc = 'top left')
plt.show()
