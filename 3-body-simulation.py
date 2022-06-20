import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

m1 = 10
m2 = 20
m3 = 30
G = 6.6743e-11
G = 9.81
total_time = 200
dt = 0.001

p1_start = np.array([-10, 10, -11])
v1_start = np.array([-3, 0, 0])

# p2_start = x_2, y_2, z_2
p2_start = np.array([0, 0, 0])
v2_start = np.array([0, 0, 0])

# p3_start = x_3, y_3, z_3
p3_start = np.array([10, 10, 12])
v3_start = np.array([3, 0, 0])

def mag(k):
    return math.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

def acceleration(p1, p2, p3):
    a1 = -G*m2*(p1-p2)/(mag(p1-p2)**3) - G*m3*(p1-p3)/(mag(p1-p3)**3)
    a2 = -G*m1*(p2-p1)/(mag(p2-p1)**3) - G*m3*(p2-p3)/(mag(p2-p3)**3)
    a3 = -G*m1*(p3-p1)/(mag(p3-p1)**3) - G*m2*(p3-p2)/(mag(p3-p2)**3)
    return a1, a2, a3

p1 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])
p2 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])
p3 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])

v1 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])
v2 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])
v3 = np.array([[0.,0.,0.] for i in range(int(total_time/dt))])

p1[0] = p1_start
p2[0] = p2_start
p3[0] = p3_start

v1[0] = v1_start
v2[0] = v2_start
v3[0] = v3_start

for i in range(len(p1)-1):
    a1, a2, a3 = acceleration(p1[i], p2[i], p3[i])
    p1[i+1] = p1[i]+v1[i]*dt
    p2[i+1] = p2[i]+v2[i]*dt
    p3[i+1] = p3[i]+v3[i]*dt
    v1[i+1] = v1[i]+a1*dt
    v2[i+1] = v2[i]+a2*dt
    v3[i+1] = v3[i]+a3*dt


# fig = plt.figure(figsize=(8, 8))
# ax = fig.gca(projection='3d')

# plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
# plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='black', lw = 0.05, markersize = 0.01, alpha=0.5)
# plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)

# plt.show()

p1 = p1.T
p2 = p2.T
p3 = p3.T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
line, = ax.plot(p1[0, 0:1], p1[1, 0:1], p1[2, 0:1])
line2, = ax.plot(p2[0, 0:1], p2[1, 0:1], p2[2, 0:1])
line3, = ax.plot(p3[0, 0:1], p3[1, 0:1], p3[2, 0:1])

# animation function.  This is called sequentially
def update(m, p1,line):
    line.set_data(p1[:2, :m*200])
    line.set_3d_properties(p1[2, :m*200])

ax.set_xlim3d([-50.0, 50.0])
ax.set_xlabel('X')

ax.set_ylim3d([-50.0, 50.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-50.0, 50.0])
ax.set_zlabel('Z')

anim = animation.FuncAnimation(fig, update, frames=10000, fargs = (p1,line), interval=1)
anim2 = animation.FuncAnimation(fig, update, frames=10000, fargs = (p2,line2), interval=1)
anim3 = animation.FuncAnimation(fig, update, frames=10000, fargs = (p3,line3), interval=1)
plt.show()

# fn = r'C:\Users\vikra\Downloads\3-body-image'
# anim.save(fn+'.gif',writer='Pillow',fps=10)