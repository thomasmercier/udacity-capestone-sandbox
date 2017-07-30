import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def make_image(theta):
    x = 32 + 25*np.sin(theta)
    y = 32 + 25*np.cos(theta)
    R = 5
    img = Image.new('L', (64,64) )
    draw = ImageDraw.Draw(img)
    draw.ellipse((x-R, y-R, x+R, y+R), fill=255)
    draw.line( (32, 32, x, y), fill=255, width=2 )
    del draw
    return np.array(img)

def image_batch(size):
    img = np.empty( (size, 64, 64, 1), dtype=np.float32 )
    theta = np.empty( (size, 1), dtype=np.float32 )
    for i in range(size):
        theta[i,0] = 2 * np.pi * np.random.rand(1)
        img[i,:,:,0] = make_image(theta[i,0])
    return [theta, img]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sess = tf.InteractiveSession()
img = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
theta = tf.placeholder(tf.float32, shape=[None, 1])

#############################
#  First convolution layer  #
#############################

size0 = 32

W_conv1 = weight_variable([4, 4, 1, size0])
b_conv1 = bias_variable([size0])
h_conv1 = tf.nn.softplus(tf.nn.conv2d(img, W_conv1, strides=[1,1,1,1], \
    padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,4,4,1], \
    strides=[1,4,4,1], padding='SAME')

#########################################################
# Fully connected layers with dimentionnality reduction #
#########################################################

h_flat1 = tf.reshape(h_pool1, [-1, 16*16*size0])

nFC = 5
W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

size1 = 64

W_fcN[0] = weight_variable([16*16*size0, size1])
b_fcN[0] = bias_variable([size1])
h_fcN[0] = tf.nn.softplus(tf.matmul(h_flat1, W_fcN[0]) + b_fcN[0])

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1])
    b_fcN[i] = bias_variable([size1])
    h_fcN[i] = tf.nn.softplus(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc2 = weight_variable([size1, 16])
b_fc2 = bias_variable([16])
h_fc2 = tf.nn.softplus(tf.matmul(h_fcN[nFC-1], W_fc2) + b_fc2)

W_fc3 = weight_variable([16, 1])
b_fc3 = bias_variable([1])
parameter = tf.matmul(h_fc2, W_fc3) + b_fc3

###########################
# training and evaluation #
###########################

loss = tf.losses.mean_squared_error(parameter, theta)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
sess.run(tf.global_variables_initializer())
M = 500

# some plotting
losstrack = np.empty(0)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.ylim([0, 10000])
#ax.set_yscale('log')
ax.set_xlim([0, M])
ax.set_ylim([1e-4, 100])
ax.set_yscale('log')
lossplt, = ax.plot([], [], 'b-')


for i in range(M):
    theta_, batch = image_batch(20)
    train_step.run(feed_dict={img: batch, theta:theta_})
    a = loss.eval(feed_dict={img: batch, theta:theta_})
    print('%i -- %f' % (i, a))

    # plot
    losstrack = np.append(losstrack, a)
    xdata = range(i+1)
    lossplt.set_xdata(xdata)
    lossplt.set_ydata(losstrack)
    fig.canvas.draw()

N = 1000
theta = np.linspace(0, 2*np.pi, N)
param = np.empty((N, 1, 1))
batch = np.empty((1, 64, 64, 1))

for i in range(N):
    batch[0,:,:,0] = make_image(theta[i])
    param[i,0,0] = parameter.eval(feed_dict={img: batch})

plt.ioff()
plt.plot(theta, param[:,0,0])
plt.show()
