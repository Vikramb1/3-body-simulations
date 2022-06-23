import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time, torch

m1 = 10
m2 = 20
m3 = 30
G = 9.8

height = 500
width = 500
max_dist = 20
total_time = 1000
dt = 0.001
divergence_distance = 150
shift_amount = 1e-3

p1_grid = np.zeros((3,height,width))
z = np.arange(20, -20, -40/width)
y = np.arange(-20, 20, 40/height)
n = np.meshgrid(y,z)
p1_grid[0] -= 10
p1_grid[1] = n[0]
p1_grid[2] = n[1]
p2_grid = np.zeros((3,height,width))
p3_grid = np.zeros((3,height,width))
p3_grid[0] += 10
p3_grid[1] += 10
p3_grid[2] += 12

n2 = n[0] + shift_amount, n[1] + shift_amount
p1_grid_prime = np.zeros((3,height,width))
p1_grid_prime[0] -= (10 - shift_amount)
p1_grid_prime[1] = n2[0]
p1_grid_prime[2] = n2[1]
p2_grid_prime = np.zeros((3,height,width))
p3_grid_prime = np.zeros((3,height,width))
p3_grid_prime[0] += 10
p3_grid_prime[1] += 10
p3_grid_prime[2] += 12

v1_grid = np.zeros(shape = (3,height,width))
v1_grid[0] -= 3
v2_grid = np.zeros(shape = (3,height,width))
v3_grid = np.zeros(shape = (3,height,width))
v3_grid[0] += 3

v1_grid_prime = np.zeros(shape = (3,height,width))
v1_grid_prime[0] -= 3
v2_grid_prime = np.zeros(shape = (3,height,width))
v3_grid_prime = np.zeros(shape = (3,height,width))
v3_grid_prime[0] += 3

ttype = torch.double

p1_grid, p2_grid, p3_grid = torch.Tensor(p1_grid), torch.Tensor(p2_grid), torch.Tensor(p3_grid)
v1_grid, v2_grid, v3_grid = torch.Tensor(v1_grid), torch.Tensor(v2_grid), torch.Tensor(v3_grid)
p1_grid_prime, p2_grid_prime, p3_grid_prime = torch.Tensor(p1_grid_prime), torch.Tensor(p2_grid_prime), torch.Tensor(p3_grid_prime)
v1_grid_prime, v2_grid_prime, v3_grid_prime = torch.Tensor(v1_grid_prime), torch.Tensor(v2_grid_prime), torch.Tensor(v3_grid_prime)

p1_grid, p2_grid, p3_grid = p1_grid.type(ttype), p2_grid.type(ttype), p3_grid.type(ttype)
v1_grid, v2_grid, v3_grid = v1_grid.type(ttype), v2_grid.type(ttype), v3_grid.type(ttype)
p1_grid_prime, p2_grid_prime, p3_grid_prime = p1_grid_prime.type(ttype), p2_grid_prime.type(ttype), p3_grid_prime.type(ttype)
v1_grid_prime, v2_grid_prime, v3_grid_prime = v1_grid_prime.type(ttype), v2_grid_prime.type(ttype), v3_grid_prime.type(ttype)

def mag(k):
    return np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

def mag_diff(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[0])**2 + (a[2] - b[2])**2)

def diverged(p1, p1_prime):
    k = np.sqrt((p1[0] - p1_prime[0])**2 + (p1[1] - p1_prime[1])**2 + (p1[2] - p1_prime[2])**2)
    return k <= divergence_distance

def acceleration(p1, p2, p3):
    a1 = -G*m2*(p1-p2)/(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**3)  - G*m3*(p1-p3)/(np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**3)
    a2 = -G*m1*(p2-p1)/(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**3) - G*m3*(p2-p3)/(np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2)**3)
    a3 = -G*m1*(p3-p1)/(np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)**3) - G*m2*(p3-p2)/(np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**3)
    return a1, a2, a3

def save(arr, i):
    plt.style.use('dark_background')
    plt.imshow(arr, cmap = 'inferno')
    plt.axis('off')
    plt.savefig('frame' + str(i) + '.png')

def save_2(arr, i, p1_grid):
    plt.style.use('dark_background')
    arr[(p1_grid[1] * 12 - p1_grid[2] * 10 < 1).numpy() & (p1_grid[1] * 12 - p1_grid[2] * 10 > -1).numpy()] = i
    plt.imshow(arr, cmap = 'inferno')
    plt.savefig('frame_proj_' + str(i) + '.png')

def save_3(arr, i):
    plt.style.use('dark_background')
    plt.imshow(arr, cmap = 'inferno')
    plt.axis('off')
    plt.savefig('frame_eject_' + str(i) + '.png')
    
def find_array(p1_grid, p2_grid, p3_grid, v1_grid, v2_grid, v3_grid, p1_grid_prime, p2_grid_prime, p3_grid_prime, v1_grid_prime, v2_grid_prime, v3_grid_prime):
    time = np.zeros((height,width))
    going_on = time < 1e10
    
    for i in range(int(total_time/dt)):
        if i%100 == 0:
            print(i)
            o = i - time
            save_3(o,i)

        distances = diverged(p1_grid,p1_grid_prime).numpy()
        going_on &= distances
        time[going_on] += 1

        a1, a2, a3 = acceleration(p1_grid, p2_grid, p3_grid)
        a1_prime, a2_prime, a3_prime = acceleration(p1_grid_prime, p2_grid, p3_grid)
        p1_grid = p1_grid + v1_grid*dt
        p2_grid = p2_grid + v2_grid*dt
        p3_grid = p3_grid + v3_grid*dt

        p1_grid_prime = p1_grid_prime + v1_grid_prime*dt
        p2_grid_prime = p2_grid_prime + v2_grid_prime*dt
        p3_grid_prime = p3_grid_prime + v3_grid_prime*dt

        v1_grid = v1_grid + a1*dt
        v2_grid = v2_grid + a2*dt
        v3_grid = v3_grid + a3*dt

        v1_grid_prime = v1_grid_prime + a1_prime*dt
        v2_grid_prime = v2_grid_prime + a2_prime*dt
        v3_grid_prime = v3_grid_prime + a3_prime*dt

        # if i%1000 == 0 and i != 0:
        #     print(a1_prime)
        #     exit()

    return time

final_plot = find_array(p1_grid, p2_grid, p3_grid, v1_grid, v2_grid, v3_grid, p1_grid_prime, p2_grid_prime, p3_grid_prime, v1_grid_prime, v2_grid_prime, v3_grid_prime)
final_plot = int(total_time/dt) - final_plot

plt.style.use('dark_background')
plt.imshow(final_plot, cmap = 'inferno')
plt.axis('off')
plt.show()
plt.close()
