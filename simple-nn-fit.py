from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_batch(size):
    n = 1
    b = 0.01
    x = 2*np.pi*n*np.random.rand(size,1)
    y = np.cos(x) + b*np.random.rand(size,1)
    return [x,y]


sess = tf.InteractiveSession()

##############################################
## Build a Multilayer Convolutional Network ##
##############################################

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# First Layer
W_conv1 = weight_variable([1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.softplus(tf.matmul(x, W_conv1) + b_conv1)

# Second Layer
W_conv2 = weight_variable([32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.softplus(tf.matmul(h_conv1, W_conv2) + b_conv2)

# 3rd Layer
W_conv3 = weight_variable([64, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.softplus(tf.matmul(h_conv2, W_conv3) + b_conv3)

# 4th Layer
W_conv4 = weight_variable([256, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.softplus(tf.matmul(h_conv3, W_conv4) + b_conv4)

# 5th Layer
W_conv5 = weight_variable([256, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.softplus(tf.matmul(h_conv3, W_conv4) + b_conv4)

# Densely Connected Layer
W_fc1 = weight_variable([256, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.softplus(tf.matmul(h_conv5, W_fc1) + b_fc1)

# Readout Layer
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# Train and Evaluate the Model
loss = tf.reduce_mean(tf.squared_difference(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())


for i in range(20000):
  batch = gen_batch(50)
  if i%100 == 0:
    train_loss = loss.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training loss %g"%(i, train_loss))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

for i in xrange(10):
    testSet = gen_batch(500)
    print("test loss %g"%loss.eval(feed_dict={x: testSet[0], y_: testSet[1]}))


n = 1
size = 1000
x2 = np.transpose([np.linspace(0, 2*np.pi*n, size)])
y2 = np.cos(x2)
y3 = y_conv.eval(feed_dict={x:x2})
plt.plot(x2, y2, x2, y3)
plt.show()
