import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image_size = 16
nFC = 5
size1 = 8
step_size = 1e-3
batch_size = 500
n_train_step = 300
encoder_dim = 2

def image_batch(size, random=True):
    L = image_size
    img = np.empty((size, L, L, 1))
    theta = np.empty((size, 1))
    if random:
        theta[:,0] = 2*np.pi*np.random.rand(size)
    else:
        theta[:,0] = np.linspace(0, 2*np.pi, num=size, endpoint=False)
    n = np.empty((size, 2)) # vector to the grey (127) line
    n[:,0] = -np.sin(theta[:,0])
    n[:,1] = np.cos(theta[:,0])
    coord_mat = np.empty((L,2,L))
    temp_one = np.ones(L)
    temp_linspace = np.linspace(1, L, num=L) - L/2.0
    coord_mat[:,0,:] = np.tensordot(temp_one, temp_linspace, axes=0)
    coord_mat[:,1,:] = np.tensordot(temp_linspace, temp_one, axes=0)
    img[:,:,:,0] = ( np.dot( n, coord_mat )*np.sqrt(2) / (L-1) + 1 ) / 2*255
    return [theta, img]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sess = tf.InteractiveSession()
img = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 1])
theta = tf.placeholder(tf.float32, shape=[None, encoder_dim])

###########
# encoder #
###########

W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(tf.nn.conv2d(img, W_conv1, strides=[1,1,1,1], \
    padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], \
    padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

h_flat = tf.reshape(h_pool2, [-1, 32*4*4])


W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

W_fcN[0] = weight_variable([32*4*4, size1])
b_fcN[0] = bias_variable([size1])
h_fcN[0] = tf.nn.relu(tf.matmul(h_flat, W_fcN[0]) + b_fcN[0])

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1])
    b_fcN[i] = bias_variable([size1])
    h_fcN[i] = tf.nn.relu(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc1 = weight_variable([size1, encoder_dim])
b_fc1 = bias_variable([encoder_dim])
encoded = tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1

###########################
# training and evaluation #
###########################

loss = tf.losses.mean_squared_error(encoded, theta)
train_step = tf.train.AdamOptimizer(step_size).minimize(loss)
sess.run(tf.global_variables_initializer())

# some plotting
losstrack = np.empty(0)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.ylim([0, 10000])
#ax.set_yscale('log')
ax.set_xlim([0, n_train_step])
ax.set_ylim([1e-5, 1e2])
ax.set_yscale('log')
lossplt, = ax.plot([], [], 'b-')


for i in range(n_train_step):
    theta_, batch = image_batch(batch_size)
    theta_sin_cos = np.empty((batch_size, 2))
    theta_sin_cos[:,0] = np.sin(theta_[:,0])
    theta_sin_cos[:,1] = np.cos(theta_[:,0])
    train_step.run(feed_dict={img: batch, theta: theta_sin_cos})
    a = loss.eval(feed_dict={img: batch, theta: theta_sin_cos})
    print('%i -- %f' % (i, a))

    # plot
    losstrack = np.append(losstrack, a)
    xdata = range(i+1)
    lossplt.set_xdata(xdata)
    lossplt.set_ydata(losstrack)
    fig.canvas.draw()

theta_, batch = image_batch(batch_size, random=False)
theta_sin_cos = np.empty((batch_size, 2))
theta_sin_cos[:,0] = np.sin(theta_[:,0])
theta_sin_cos[:,1] = np.cos(theta_[:,0])
output_ = encoded.eval(feed_dict={img: batch, theta: theta_sin_cos})

plt.ioff()
ax2 = fig.add_subplot(222)
ax2.plot(theta_, output_[:,0])
ax2.plot(theta_, output_[:,1])
plt.show()
