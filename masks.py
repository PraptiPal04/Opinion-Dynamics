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
# A=dy.small_world_network(N,2)
# #x0=(np.random.uniform(-1.0,1.0,size=N))
# x0=(np.random.uniform(-1.0,1.0,size=N))
# #x0=np.append(x0,1)

A=dy.small_world_network_with_rand(N,0.25,2)
x0=(np.random.uniform(-1.0,1.0,size=N))

# b=np.array([0.2,0.,0.,0.,0.,0.,0.,0.,0.,0.])
# x,t,h,err,d,u = rkf_var(dy.rhs,x0,0.05,N,1,0.15,dyn_d,undyn_u,A=A,al=1,gm=1.3,b=b) 
# print(A)
# print(x0)

# for i in range(N):
#     plt.plot(t,x[i,:])

#plt.plot(t,u)

# gr.graph_animate(A,x)

# plt.show()

