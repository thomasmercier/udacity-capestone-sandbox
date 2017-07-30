import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image_size = 8

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
theta = tf.placeholder(tf.float32, shape=[None, 2])

nFC = 5
W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

size1 = 64

img_flat = tf.reshape(img, [-1, image_size*image_size])
W_fcN[0] = weight_variable([image_size*image_size, size1])
b_fcN[0] = bias_variable([size1])
h_fcN[0] = tf.nn.relu(tf.matmul(img_flat, W_fcN[0]) + b_fcN[0])

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1])
    b_fcN[i] = bias_variable([size1])
    h_fcN[i] = tf.nn.relu(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc2 = weight_variable([size1, 2])
b_fc2 = bias_variable([2])
output = tf.matmul(h_fcN[nFC-1], W_fc2) + b_fc2

###########################
# training and evaluation #
###########################

print(1111111111111111111111111111111111111111111)
print(output)
print(theta)
print(22222222222222222222222222222222222222222)
loss = tf.losses.mean_squared_error(output, theta)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
sess.run(tf.global_variables_initializer())
M = 300

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
    theta_, batch = image_batch(10)
    theta__ = np.empty((len(theta_[:,0]), 2)) # vector to the grey (127) line
    theta__[:,0] = np.cos(theta_[:,0])
    theta__[:,1] = np.sin(theta_[:,0])
    train_step.run(feed_dict={img: batch, theta:theta__})
    a = loss.eval(feed_dict={img: batch, theta:theta__})
    print('%i -- %f' % (i, a))

    # plot
    losstrack = np.append(losstrack, a)
    xdata = range(i+1)
    lossplt.set_xdata(xdata)
    lossplt.set_ydata(losstrack)
    fig.canvas.draw()

N = 100
theta_, batch = image_batch(N, random=False)
output_ = output.eval(feed_dict={img: batch})

for i in range(N):
    tmp = batch[i,:,:,:]
    print( '%i -- %f -- %f'%(i, np.amax(tmp), np.amin(tmp)) )

print(output_.shape)

plt.ioff()
ax2 = fig.add_subplot(222)
ax2.plot(theta_, output_[:,0])
ax2.plot(theta_, output_[:,1])
plt.show()
