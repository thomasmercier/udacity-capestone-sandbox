import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image_size = 8
nFC = 50
size1 = 16
nFC2 = 50
step = 10
M = 300
beta = 0.01
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

###########
# encoder #
###########

W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

img_flat = tf.reshape(img, [-1, image_size*image_size])
W_fcN[0] = weight_variable([image_size*image_size, size1])
regularizer_loss = tf.nn.l2_loss(W_fcN[0])
b_fcN[0] = bias_variable([size1])
regularizer_loss += tf.nn.l2_loss(b_fcN[0])
h_fcN[0] = tf.nn.tanh(tf.matmul(img_flat, W_fcN[0]) + b_fcN[0])

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1])
    regularizer_loss += tf.nn.l2_loss(W_fcN[i])
    b_fcN[i] = bias_variable([size1])
    regularizer_loss += tf.nn.l2_loss(b_fcN[i])
    h_fcN[i] = tf.nn.tanh(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc1 = weight_variable([size1, encoder_dim])
regularizer_loss += tf.nn.l2_loss(W_fc1)
b_fc1 = bias_variable([encoder_dim])
regularizer_loss += tf.nn.l2_loss(b_fc1)
encoded = tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1

###########
# decoder #
###########

W_fc2 = weight_variable([encoder_dim, size1])
regularizer_loss += tf.nn.l2_loss(W_fc2)
b_fc2 = bias_variable([size1])
regularizer_loss += tf.nn.l2_loss(b_fc2)
h_fc2 = tf.matmul(encoded, W_fc2) + b_fc2

W_fcM = [None]*nFC2
b_fcM = [None]*nFC2
h_fcM = [None]*nFC2

W_fcM[0] = weight_variable([size1, size1])
regularizer_loss += tf.nn.l2_loss(W_fcM[0])
b_fcM[0] = bias_variable([size1])
regularizer_loss += tf.nn.l2_loss(b_fcM[0])
h_fcM[0] = tf.nn.tanh(tf.matmul(h_fc2, W_fcM[0]) + b_fcM[0])

for i in range(1, nFC2):
    W_fcM[i] = weight_variable([size1, size1])
    regularizer_loss += tf.nn.l2_loss(W_fcM[i])
    b_fcM[i] = bias_variable([size1])
    regularizer_loss += tf.nn.l2_loss(b_fcM[i])
    h_fcM[i] = tf.nn.tanh(tf.matmul(h_fcM[i-1], W_fcM[i]) + b_fcM[i])

W_fc3 = weight_variable([size1, image_size*image_size])
regularizer_loss += tf.nn.l2_loss(W_fc3)
b_fc3 = bias_variable([image_size*image_size])
regularizer_loss += tf.nn.l2_loss(b_fc3)
h_fc3 = tf.nn.tanh(tf.matmul(h_fcM[nFC2-1], W_fc3) + b_fc3)

W_fc4 = weight_variable([image_size*image_size, image_size*image_size])
regularizer_loss += tf.nn.l2_loss(W_fc4)
b_fc4 = bias_variable([image_size*image_size])
regularizer_loss += tf.nn.l2_loss(b_fc4)
h_fc4 = tf.nn.tanh(tf.matmul(h_fc3, W_fc4) + b_fc4)

decoded = tf.reshape(h_fc4, (-1, image_size, image_size, 1))

###########################
# training and evaluation #
###########################

obj_loss = tf.losses.mean_squared_error(decoded, img)
loss = obj_loss + beta*regularizer_loss
train_step = tf.train.AdamOptimizer(step).minimize(loss)
sess.run(tf.global_variables_initializer())

# some plotting
losstrack = np.empty(0)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.ylim([0, 10000])
#ax.set_yscale('log')
ax.set_xlim([0, M])
ax.set_ylim([1e0, 1e6])
ax.set_yscale('log')
lossplt, = ax.plot([], [], 'b-')


for i in range(M):
    theta_, batch = image_batch(10)
    train_step.run(feed_dict={img: batch})
    a = obj_loss.eval(feed_dict={img: batch})
    b = beta*regularizer_loss.eval(feed_dict={img: batch})
    print('%i -- %f -- %f' % (i, a, b))

    # plot
    losstrack = np.append(losstrack, a)
    xdata = range(i+1)
    lossplt.set_xdata(xdata)
    lossplt.set_ydata(losstrack)
    fig.canvas.draw()

N = 100
theta_, batch = image_batch(N, random=False)
output_ = encoded.eval(feed_dict={img: batch})

for i in range(N):
    tmp = batch[i,:,:,:]

plt.ioff()
ax2 = fig.add_subplot(222)
ax2.plot(theta_, output_[:,0])
ax2.plot(theta_, output_[:,1])
plt.show()
