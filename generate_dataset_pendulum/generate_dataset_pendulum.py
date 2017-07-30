import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



# pendulum physical parameters
l = 1
g = 9.81
m = 1
Q = 20

# simulation parameters
dt = 1e-2
duration = 20
x0 = [0,0]
u0 = 10

# input parameters
u0 = 100
omega1 = 0.01
omega2 = 20
nSin = 50

# output parameters
height = 64
width = 64
l_px = 25
l_width = 2
mass_radius = 5
out_dir = 'data/'

# useful quantities
N = int(duration/dt)
t = dt*np.array(range(N))

def input():
    result = np.zeros(N)
    for i in range(nSin):
        omega = ( i*omega2 + (nSin-i-1)*omega1 ) / float(nSin-1)
        ampl = u0 / float(nSin)
        phi = 2 * np.pi * np.random.rand(1)
        result += ampl * np.cos( omega*t + phi )
    return result

def f(x,u):
    I = float(m)*l**2
    omega = np.sqrt(float(g)/l)
    temp = - omega**2*np.sin(x[0]) - omega/Q*x[1] + u/I
    return np.array( (x[1], temp) )

def simulate(u):
    x = np.empty((N,2), dtype=float)
    x[0,:] = x0
    for i in range(N-1):
        x[i+1,:] = x[i,:] + dt*f( x[i,:], u[i] )
    #plt.plot(t, x[:,0])
    #plt.plot(t, u)
    #plt.show()
    return x

def make_image(theta):
    w = width/2
    h = height/2
    x = w + l_px*np.sin(theta)
    y = h + l_px*np.cos(theta)
    R = mass_radius
    img = Image.new('L', (width,height) )
    draw = ImageDraw.Draw(img)
    draw.ellipse((x-R, y-R, x+R, y+R), fill=255)
    draw.line( (w, h, x, y), fill=255, width=l_width )
    del draw
    return img

u = input()
x = simulate(u)
np.savetxt(out_dir+'input.txt', u)
for i in range(N):
    imgFile = out_dir + str(i) + '.png'
    img = make_image(x[i,0])
    img.save(imgFile, 'PNG')
