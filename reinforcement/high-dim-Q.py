import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import tensorflow as tf
import random

def dist(P, A, u):

    proj = np.dot(P-A,u)
    return np.linalg.norm(P-A-proj*u)

def levelset(P):

    v1 = 1
    v2 = 2
    n = len(P)
    t = np.zeros(n)
    A = np.zeros(n)
    t[0] = 1
    B = np.zeros(n)
    B[0] = 1
    u = np.zeros(n)
    u[1] = 1
    R = 0.25
    if dist(P, A, t) > R and dist(P, B, u) > R:
        return v1
    else:
        return v2

def test():

    N = 30
    xy = np.empty((N*N,2))
    z = np.empty(N*N)
    for i in range(N):
        for j in range(N):
            xy[i+N*j,:] = (float(i)/N*2-0.5,float(j)/N*2-0.5)
            z[i+N*j] = levelset(xy[i+N*j,:])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xy[:,0], xy[:,1], z, 'g.')
    plt.show()

def transition(state, action):

    v1 = 1
    v2 = 2
    norm = np.norm(action)
    if norm == 0:
        return x
    else:
        return state + levelset(state) / norm * action

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class QNet:

    def __init__(self, state_dim, action_dim, step_size):

        nFC = 30
        size1 = 10
        size2 = 10

        self.state = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.action = tf.placeholder(tf.float32, shape=[None, action_dim])
        self.Qobjective = tf.placeholder(tf.float32, shape=[None, 1])

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
        self.Q = tf.nn.sigmoid(tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1)
        self.loss = tf.losses.mean_squared_error(self.Qobjective, self.Q)
        self.train_step = tf.train.AdamOptimizer(step_size).minimize(loss)

class MuNet:

    def __init__(state_dim):

        nFC = 30
        size1 = 10
        size2 = 10

        self.state = tf.placeholder(tf.float32, shape=[None, state_dim])

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
        self.action = tf.nn.sigmoid(tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1)

class ReplayBuffer:

    def __init__(self, state_size, action_size, size):

        self.entries = { 'state': np.empty((0, state_dim)), \
                         'action': np.empty((0, action_dim)), \
                         'reward': np.empty(0), \
                         'next_state': np.empty((0, state_dim)) }
        self.size = size


    def add(state, action, reward, next_state, new_entry):

        if len(self.state) > self.size:
            for key in self.entries:
                self.entries[key] = np.roll(self.entries[key], -1, axis=0)
                self.entries[self.size-1] = new_entry[key]
        else:
            for key in self.entries:
                self.entries[key].append(new_entry[key])

    def sample(size):

        idx = random.range()
        return random.sample(self.entries, size)




state_dim = 2
action_dim = 2
n_episode = 10
n_replay = 10
n_replay_sample = 3
max_time = 10
eps = 0.1
discount_factor = 0.99
target_state = np.zeros(state_dim)
target_state[0:1] = [1, 1]
target_radius = 0.25

buff = ReplayBuffer(state_dim, action_dim, size)

sess = tf.InteractiveSession()
QNet1 = QNet(state_dim, action_dim)
QNet2 = QNet1.copy()
muNet1 = muNet(state_dim)
muNet2 = muNet1.copy()
sess.run(tf.global_variables_initializer())

for i_episode in range(n_episode):
    state = np.zeros(state_dim)
    for i_step in range(max_time):
        if np.random.rand(1) < eps:
            action = np.random.rand(action_dim)
        else:
            action = muNet2.action.eval(feed_dict={self.state: state})
        next_state = transition(state, action)
        if np.norm(next_state-target_state) < target_radius:
            reward = 1
        else:
            reward = -0.01
        buff.add(state, action, reward, next_state);
        sample = buff.sample(n_replay_sample)
        mu1 = muNet1.action.eval(feed_dict={muNet1.state: sample)
        Q1 = QNet1.Q.eval(feed_dict={QNet1.state: next_state, QNet1.action: mu1})
        Qobjective = sample['reward'] + discount_factor * Q1
        QNet2.train_step.run(feed_dict={img: batch})
