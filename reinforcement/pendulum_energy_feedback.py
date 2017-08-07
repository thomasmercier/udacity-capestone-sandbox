import numpy as np
import matplotlib.pyplot as plt

m = 1
Q = 10
L = 1
g = 9.81
uMax = 10
N = 3000
dt = 0.001
x0 = 0
v0 = 5


I = m*L**2
w0 = np.sqrt(g/L)

def f(x, u):
    res1 = ( u - w0/Q*x[1] - w0**2*np.sin(x[0]) ) / I
    return np.array([x[1], res1])

def u(x):
    E = 0.5*I*x[1]**2 - m*g*L*np.cos(x[0])
    dW = E - m*g*L
    print(dW)
    return -np.sign(dW) * np.sign(x[1]) * uMax;

x = np.empty([N, 2])
x[0,0] = x0
x[0,1] = v0

for i in range(N-1):
    x[i+1,:] = x[i] + dt*f(x[i,:], u(x[i,:]))

plt.plot(x[:,0])
plt.show()
