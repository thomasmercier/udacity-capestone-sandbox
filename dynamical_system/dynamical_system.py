import numpy as np
import matplotlib.pyplot as plt


def f(x,u):
    l = 1
    g = 9.81
    m = 1
    Q = 20
    I = float(m)*l**2
    omega = np.sqrt(float(g)/l)
    temp = - omega**2*np.sin(x[0]) - omega/Q*x[1] + u/I
    return np.array( (x[1], temp) )

def simulate():
    dt = 1e-3
    duration = 20
    x0 = [0,1]
    u0 = 10



    N = int(duration/dt)
    t = dt*np.array(range(N))
    x = np.empty((N,2), dtype=float)
    x[0,:] = x0
    u = u0*np.ones_like(t)
    for i in range(N-1):
        x[i+1,:] = x[i,:] + dt*f( x[i,:], u[i] )
    plt.plot(t, x[:,1])
    plt.show()

simulate()
