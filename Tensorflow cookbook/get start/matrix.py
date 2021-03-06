# Matrix and Matrix Operation

import numpy as np 
import tensorflow as tf 

from tensorflow.python.framework import ops
ops.reset_default_graph()

# Declaring a matrix
sess = tf.Session()

# Identity matrix
identity_matrix = tf.diag([1.0, 1.0, 1.0])
print(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
print(sess.run(A))

# 2x3 constant matrix
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 3x2 random uniform matrix
C = tf.random_uniform([3,2])
print(sess.run(C))
print(sess.run(C))

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/substraction
print(sess.run(A+B))
print(sess.run(A-B))

# Matrix Multiplication
print(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose
print(sess.run(tf.transpose(C)))

# Matrix Determinant
print(sess.run(tf.matrix_determinant(D)))

# Matrix Inverse
print(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors
print(sess.run(tf.self_adjoint_eig(D)))