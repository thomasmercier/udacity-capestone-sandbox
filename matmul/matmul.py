import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
d = np.array([[1, 0], [0, 2]], dtype=np.float32)
print(a)
b = np.empty((1, 2, 3))
b[0, :, :, 0] = a
f = np.empty((1, 2, 2))
f[0, :, :, 0] = d
c = tf.constant(a)
g = tf.constant(d)
h = tf.matmul(g, c)
print(c.eval())
print(g.eval())
print(h.eval())
