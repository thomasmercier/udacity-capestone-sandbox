import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image_size = 16
nFC = 2
size1 = 16
step_size = 1e-3
batch_size = 500
n_train_step = 1000
encoder_dim = 2
alpha = 100

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

def upsampling2d(in_layer, size):
    in_shape = in_layer.shape.as_list()
    #a = np.ones(size, dtype=np.float32)
    a = np.zeros((size[0], size[1], in_shape[3], in_shape[3]), dtype=np.float32)
    for i in range(in_shape[3]):
        a[:,:,i,i] = np.ones(size)
    a = tf.constant(a)
    output_shape = (batch_size, in_shape[1]*size[0], \
                                 in_shape[2]*size[1], in_shape[3])
    strides = (1, size[0], size[1], 1)
    return tf.nn.conv2d_transpose(in_layer, a, output_shape, strides)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()
img = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])

###########
# encoder #
###########

enc_W_conv1 = weight_variable([3, 3, 1, 16])
enc_b_conv1 = bias_variable([16])
enc_h_conv1 = tf.nn.relu(tf.nn.conv2d(img, enc_W_conv1, strides=[1,1,1,1], \
    padding='SAME') + enc_b_conv1)
enc_h_pool1 = tf.nn.max_pool(enc_h_conv1, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

enc_W_conv2 = weight_variable([3, 3, 16, 32])
enc_b_conv2 = bias_variable([32])
enc_h_conv2 = tf.nn.relu(tf.nn.conv2d(enc_h_pool1, enc_W_conv2, strides=[1,1,1,1], \
    padding='SAME') + enc_b_conv2)
enc_h_pool2 = tf.nn.max_pool(enc_h_conv2, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

enc_h_flat = tf.reshape(enc_h_pool2, [-1, 32*4*4])


enc_W_fcN = [None]*nFC
enc_b_fcN = [None]*nFC
enc_h_fcN = [None]*nFC

enc_W_fcN[0] = weight_variable([32*4*4, size1])
enc_b_fcN[0] = bias_variable([size1])
enc_h_fcN[0] = tf.nn.relu(tf.matmul(enc_h_flat, enc_W_fcN[0]) + enc_b_fcN[0])

for i in range(1, nFC):
    enc_W_fcN[i] = weight_variable([size1, size1])
    enc_b_fcN[i] = bias_variable([size1])
    enc_h_fcN[i] = tf.nn.relu(tf.matmul(enc_h_fcN[i-1], enc_W_fcN[i]) + enc_b_fcN[i])

enc_W_fc1 = weight_variable([size1, encoder_dim])
enc_b_fc1 = bias_variable([encoder_dim])
encoded = tf.matmul(enc_h_fcN[nFC-1], enc_W_fc1) + enc_b_fc1

###########
# decoder #
###########

dec_W_fcN = [None]*nFC
dec_b_fcN = [None]*nFC
dec_h_fcN = [None]*nFC

dec_W_fcN[0] = weight_variable([encoder_dim, size1])
dec_b_fcN[0] = bias_variable([size1])
dec_h_fcN[0] = tf.matmul(encoded, dec_W_fcN[0]) + dec_b_fcN[0]

for i in range(1, nFC):
    dec_W_fcN[i] = weight_variable([size1, size1])
    dec_b_fcN[i] = bias_variable([size1])
    dec_h_fcN[i] = tf.nn.relu(tf.matmul(dec_h_fcN[i-1], dec_W_fcN[i]) + dec_b_fcN[i])

dec_W_fc1 = weight_variable([size1, 64*2*2])
dec_b_fc1 = bias_variable([64*2*2])
dec_h_fc1 = tf.matmul(dec_h_fcN[nFC-1], dec_W_fc1) + dec_b_fc1
dec_h_mat1 = tf.reshape(dec_h_fc1, (-1, 2, 2, 64))


dec_h_upsamp1 = upsampling2d(dec_h_mat1, (2, 2))
dec_W_conv1 = weight_variable([3, 3, 64, 64])
dec_b_conv1 = bias_variable([64])
dec_h_conv1 = tf.nn.relu(tf.nn.conv2d(dec_h_upsamp1, dec_W_conv1, strides=[1,1,1,1], \
    padding='SAME') + dec_b_conv1)

dec_h_upsamp2 = upsampling2d(dec_h_conv1, (2,2))
dec_W_conv2 = weight_variable([3, 3, 64, 16])
dec_b_conv2 = bias_variable([16])
dec_h_conv2 = tf.nn.relu(tf.nn.conv2d(dec_h_upsamp2, dec_W_conv2, strides=[1,1,1,1], \
    padding='SAME') + dec_b_conv2)

dec_h_upsamp3 = upsampling2d(dec_h_conv2, (2,2))
dec_W_conv3 = weight_variable([3, 3, 16, 1])
dec_b_conv3 = bias_variable([1])
decoded = tf.nn.relu(tf.nn.conv2d(dec_h_upsamp3, dec_W_conv3, strides=[1,1,1,1], \
    padding='SAME') + dec_b_conv3)




###########################
# training and evaluation #
###########################

loss_reconstruction = tf.losses.mean_squared_error(decoded, img)
loss_dim_reduction = (tf.nn.l2_loss(encoded) / batch_size*2 - 1)**2
loss = loss_reconstruction + alpha*loss_dim_reduction
train_step = tf.train.AdamOptimizer(step_size).minimize(loss)
sess.run(tf.global_variables_initializer())

def plot_training(fig, location, n_train_step, ylim):
    ax = fig.add_subplot(111)
    ax.set_xlim([0, n_train_step])
    ax.set_ylim([1e-2, 1e6])
    ax.set_yscale('log')
    lossplt, = ax.plot([], [], 'b-')
    return lossplt


# some plotting
losstrack = np.empty(0)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_xlim([0, n_train_step])
#ax.set_ylim([1e-2, 1e6])
ax.set_yscale('log')
lossplt, = ax.plot([], [], 'b-')


for i in range(n_train_step):
    theta_, batch = image_batch(batch_size)
    train_step.run(feed_dict={img: batch})
    a = loss_reconstruction.eval(feed_dict={img: batch})
    b = loss_dim_reduction.eval(feed_dict={img: batch})
    print('%i -- %f -- %f' % (i, a, b))

    # plot
    losstrack = np.append(losstrack, a)
    xdata = range(i+1)
    lossplt.set_xdata(xdata)
    lossplt.set_ydata(losstrack)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

theta_, batch = image_batch(batch_size, random=False)
output_ = encoded.eval(feed_dict={img: batch})

plt.ioff()
ax2 = fig.add_subplot(222)
ax2.plot(theta_, output_[:,0])
ax2.plot(theta_, output_[:,1])
plt.show()
