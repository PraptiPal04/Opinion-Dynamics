# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:10:22 2021

@author: Prapti
"""


import numpy as np
import matplotlib.pyplot as plt
from RKF import rkf_var
import dynamics as dy
import graph_plot as gr

def dyn_u(u0,t):
    if u0<0.5:
        u= u0 + 0.005
    else:
        u=u0
    return u

def dyn_d(d0,t):
    if d0>0.25:
        d=d0-0.01
    else:
        d=d0
    return d

def undyn_d(d0,t):
    return d0

def undyn_u(u0,t):
    return u0
        
N=10
#A=dy.small_world_network(N,2)
# #x0=(np.random.uniform(-1.0,1.0,size=N))
# x0=(np.random.uniform(-1.0,1.0,size=N))
# #x0=np.append(x0,1)

A=dy.small_world_network_with_rand(N,0.25,2)
x0=(np.random.uniform(-1.0,1.0,size=N))


# stuff for smalla world prob 0.25 one bias 0.2 dyn u IMP
# A = np.array([[0., 0., 1., 1., 0., 0., 1., 0., 1., 1.],[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],[1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 1., 0., 0., 0., 1.],[0., 0., 0., 0., 0., 1., 1., 0., 1., 0.],[0., 0., 1., 1., 1., 0., 0., 1., 0., 0.],[1., 0., 0., 0., 1., 0., 0., 1., 1., 0.],[0., 0., 0., 0., 0., 1., 1., 0., 1., 1.],[1., 1., 0., 0., 1., 0., 1., 1., 0., 1.],[1., 1., 0., 1., 0., 0., 0., 1., 1., 0.]])
# x0 = [-0.8404353,0.49153123,-0.64393996,-0.69269768,-0.51168166,0.4420671,0.91784232,-0.61532467,0.3910434,-0.27834121]

# stuff for smalla world prob 0.25 one bias 0.2 dyn d dyn u IMP
# A = np.array([[0., 0., 1., 1., 0., 0., 0., 0., 1., 1.],[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],[1., 0., 0., 0., 0., 0., 0., 0., 1., 1.],[1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],[0., 0., 0., 1., 0., 1., 1., 0., 0., 1.],[0., 0., 0., 0., 1., 0., 1., 1., 0., 1.],[0., 1., 0., 0., 1., 1., 0., 1., 1., 1.],[0., 0., 0., 1., 0., 1., 1., 0., 1., 0.],[1., 0., 1., 0., 0., 0., 1., 1., 0., 0.],[1., 1., 1., 0., 1., 1., 1., 0., 0., 0.]])
# x0 = [ 0.22893641, -0.26576308, -0.23562549,  0.07997758,  0.47455421, -0.99429337, 0.18908439, -0.78121144,  0.28010946, -0.11828395]

b=np.array([0.2,0.,0.,0.,0.,0.,0.,0.,0.,0.])
x,t,h,err,d,u = rkf_var(dy.rhs,x0,0.05,N,1,0.15,dyn_d,undyn_u,A=A,al=1,gm=1.3,b=b) 
print(A)
print(x0)

# for i in range(N):
#     plt.plot(t,x[i,:])
# plt.xlabel("t")
# plt.ylabel("x")
# plt.ylim(-1.2,1.)
# plt.title("Small_worlds prob : 0.25 with bias : 0.2 dyn d + dyn u")
# plt.show()
#plt.plot(t,u)

gr.graph_animate(A,x)

