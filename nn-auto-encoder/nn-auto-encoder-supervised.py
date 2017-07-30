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

W_conv1 = weight_variable([7, 7, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.softplus(tf.nn.conv2d(img, W_conv1, strides=[1,1,1,1], \
    padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

############################
#  First inception module  #
############################

# 3x3

W_conv2_1 = weight_variable([1, 1, 16, 4])
b_conv2_1 = bias_variable([4])
h_conv2_1 = tf.nn.softplus(tf.nn.conv2d(h_pool1, W_conv2_1, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_1)
W_conv2_2 = weight_variable([3, 3, 4, 4])
b_conv2_2 = bias_variable([4])
h_conv2_2 = tf.nn.softplus(tf.nn.conv2d(h_conv2_1, W_conv2_2, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_2)
W_conv2_3 = weight_variable([1, 1, 4, 16])
b_conv2_3 = bias_variable([16])
h_conv2_3 = tf.nn.softplus(tf.nn.conv2d(h_conv2_2, W_conv2_3, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_3)

# 5x5

W_conv2_4 = weight_variable([1, 1, 16, 4])
b_conv2_4 = bias_variable([4])
h_conv2_4 = tf.nn.softplus(tf.nn.conv2d(h_pool1, W_conv2_4, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_4)
W_conv2_5 = weight_variable([5, 5, 4, 4])
b_conv2_5 = bias_variable([4])
h_conv2_5 = tf.nn.softplus(tf.nn.conv2d(h_conv2_4, W_conv2_5, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_5)
W_conv2_6 = weight_variable([1, 1, 4, 16])
b_conv2_6 = bias_variable([16])
h_conv2_6 = tf.nn.softplus(tf.nn.conv2d(h_conv2_5, W_conv2_6, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_6)

# pooling + 1x1

h_pool2 = tf.nn.max_pool(h_pool1, ksize=[1,3,3,1], \
    strides=[1,1,1,1], padding='SAME')
W_conv2_7 = weight_variable([1, 1, 16, 16])
b_conv2_7 = bias_variable([16])
h_conv2_7 = tf.nn.softplus(tf.nn.conv2d(h_pool2, W_conv2_7, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_7)

# 1x1

W_conv2_8 = weight_variable([1, 1, 16, 16])
b_conv2_8 = bias_variable([16])
h_conv2_8 = tf.nn.softplus(tf.nn.conv2d(h_pool1, W_conv2_8, strides=[1,1,1,1], \
    padding='SAME') + b_conv2_8)

# Depth concatenation

h_concat1 = tf.concat( [h_conv2_3, h_conv2_6, h_conv2_7, h_conv2_8], 3)


#############################
#  Second inception module  #
#############################

# 3x3

W_conv3_1 = weight_variable([1, 1, 64, 16])
b_conv3_1 = bias_variable([16])
h_conv3_1 = tf.nn.softplus(tf.nn.conv2d(h_concat1, W_conv3_1, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_1)
W_conv3_2 = weight_variable([3, 3, 16, 16])
b_conv3_2 = bias_variable([16])
h_conv3_2 = tf.nn.softplus(tf.nn.conv2d(h_conv3_1, W_conv3_2, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_2)
W_conv3_3 = weight_variable([1, 1, 16, 64])
b_conv3_3 = bias_variable([64])
h_conv3_3 = tf.nn.softplus(tf.nn.conv2d(h_conv3_2, W_conv3_3, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_3)

# 5x5

W_conv3_4 = weight_variable([1, 1, 64, 16])
b_conv3_4 = bias_variable([16])
h_conv3_4 = tf.nn.softplus(tf.nn.conv2d(h_concat1, W_conv3_4, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_4)
W_conv3_5 = weight_variable([5, 5, 16, 16])
b_conv3_5 = bias_variable([16])
h_conv3_5 = tf.nn.softplus(tf.nn.conv2d(h_conv3_4, W_conv3_5, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_5)
W_conv3_6 = weight_variable([1, 1, 16, 64])
b_conv3_6 = bias_variable([64])
h_conv3_6 = tf.nn.softplus(tf.nn.conv2d(h_conv3_5, W_conv3_6, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_6)

# pooling + 1x1

h_pool3 = tf.nn.max_pool(h_concat1, ksize=[1,3,3,1], \
    strides=[1,1,1,1], padding='SAME')
W_conv3_7 = weight_variable([1, 1, 64, 64])
b_conv3_7 = bias_variable([64])
h_conv3_7 = tf.nn.softplus(tf.nn.conv2d(h_pool3, W_conv3_7, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_7)

# 1x1

W_conv3_8 = weight_variable([1, 1, 64, 64])
b_conv3_8 = bias_variable([64])
h_conv3_8 = tf.nn.softplus(tf.nn.conv2d(h_concat1, W_conv3_8, strides=[1,1,1,1], \
    padding='SAME') + b_conv3_8)

# Depth concatenation

h_concat2 = tf.concat( [h_conv3_3, h_conv3_6, h_conv3_7, h_conv3_8], 3)
h_pool4 = tf.nn.max_pool(h_concat1, ksize=[1,2,2,1], \
    strides=[1,2,2,1], padding='SAME')

#########################################################
# Fully connected layers with dimentionnality reduction #
#########################################################

h_flat1 = tf.reshape(h_pool4, [-1, 16*16*64])

W_fc1 = weight_variable([16*16*64, 64])
b_fc1 = bias_variable([64])
h_fc1 = tf.nn.softplus(tf.matmul(h_flat1, W_fc1) + b_fc1)

W_fc2 = weight_variable([64, 64])
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.softplus(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([64, 1])
b_fc3 = bias_variable([1])
parameter = tf.matmul(h_fc2, W_fc3) + b_fc3

###########################
# training and evaluation #
###########################

print(theta)
print(parameter)
loss = tf.losses.mean_squared_error(parameter, theta)
h_pool1_norm = tf.norm(h_pool1)
h_concat1_norm = tf.norm(h_concat1)
h_concat2_norm = tf.norm(h_concat2)

train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
sess.run(tf.global_variables_initializer())
for i in range(1):
    theta_, batch = image_batch(100)
    train_step.run(feed_dict={img: batch, theta:theta_})
    a = loss.eval(feed_dict={img: batch, theta:theta_})
    b = h_pool1_norm.eval(feed_dict={img: batch, theta:theta_})
    c = h_concat1_norm.eval(feed_dict={img: batch, theta:theta_})
    d = h_concat2_norm.eval(feed_dict={img: batch, theta:theta_})

    print('%i -- %f -- %f -- %f -- %f' % (i, a, b, c, d))

N = 10
theta = np.linspace(0, 2*np.pi, N)
param = [None]*N
batch = np.empty((1, 64, 64, 1))
for i in range(N):
    batch[0,:,:,0] = make_image(theta[i])
    param[i] = parameter.eval(feed_dict={img: batch})
    print(param[i])
    print(33333333333333333333333333333333333333333333)
plt.plot(theta, param[:,0,0])
plt.show()
