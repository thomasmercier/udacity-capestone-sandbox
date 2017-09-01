from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


nFC = 30
size1 = 10
size2 = 10
step_size = 1e-3
n_train_step = 100
batch_size = 1000


def f(x,u):
    return (u+np.power(u,2))*np.sinc(u*x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 1])
u = tf.placeholder(tf.float32, shape=[None, 1])
fxu = tf.placeholder(tf.float32, shape=[None, 1])

W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

W_fcN[0] = weight_variable([1, size1])
b_fcN[0] = bias_variable([size1])
h_fcN[0] = tf.matmul(x, W_fcN[0]) + b_fcN[0]

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1])
    b_fcN[i] = bias_variable([size1])
    h_fcN[i] = tf.nn.sigmoid(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc1 = weight_variable([size1, 2*size2])
b_fc1 = bias_variable([2*size2])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1)

yi, ui = tf.split(h_fc1, [size2, size2], 1)

wire_range = tf.reduce_max(yi, axis=1, keep_dims=True) \
                - tf.reduce_min(yi, axis=1, keep_dims=True)
weight = tf.reciprocal(tf.abs(ui-u) + wire_range)

estimate = tf.reduce_sum(yi*weight, axis=1, keep_dims=True) \
                / tf.reduce_sum(weight, axis=1, keep_dims=True)
loss = tf.losses.mean_squared_error(estimate, fxu)
train_step = tf.train.AdamOptimizer(step_size).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(n_train_step):
    x_ = np.random.rand(batch_size,1)
    u_ = np.random.rand(batch_size,1)
    fxu_ = f(x_,u_)
    train_step.run(feed_dict={x: x_, u: u_, fxu: fxu_})
    a = loss.eval(feed_dict={x: x_, u: u_, fxu: fxu_})
    print('%i -- %f' % (i, a))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1):
    #x_ = np.empty((size2, 1))
    #u_ = np.empty((size2, 1))
    #x_[:,0] = np.linspace(0.0, 1.0, num=size2)
    #u_[:,0] = np.linspace(0.0, 1.0, num=size2)
    x_ = np.random.rand(batch_size,1)
    u_ = np.random.rand(batch_size,1)
    fxu_ = estimate.eval(feed_dict={x: x_, u: u_})
    ui_ = ui.eval(feed_dict={x: x_, u: u_})
    yi_ = yi.eval(feed_dict={x: x_, u: u_})
    ax.scatter(x_[:,0], ui_[:,0], yi_[:,0], 'g.')
    ax.scatter(x_[:,0], u_[:,0], fxu_[:,0], 'r.')
    ax.scatter(x_[:,0], u_[:,0], f(x_,u_)[:,0], 'b.')

plt.show()
