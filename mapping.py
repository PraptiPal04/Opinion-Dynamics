# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:23:31 2021

@author: Prapti
"""


import numpy as np
import matplotlib.pyplot as plt
from RKF import rkf
import dynamics as dy
import networkx as nx
import matplotlib as mpl
import matplotlib.animation as an
import graph_plot as gr


def cascade(A,x):
    '''
    Checks if a particular opinion network is in Opinion Casade.

    Parameters
    ----------
    A : numpy array
        adjacency matrix of the graph
    x : numpy array
        opinions of each node of the network

    Returns
    -------
    int
        Either 1 or 0. State 1 corresponds to opinion cascade. State 0 
        corresponds to no cascade.

    '''
    
    row,col = np.where(A==1.)
    flag = 1
    for i in range(len(row)):
        if (np.sign(x[row[i]])==np.sign(x[col[i]])):
            flag=0
    if (flag==1):
        return 1
    else:
        return 0

def map_alpha_gamma(func,*args,**kwargs):
    
    al=np.arange(-1.0,8,0.1)
    gm=np.arange(-6.0,6,0.1)
    X = np.zeros((N,len(al),len(gm)))
    for i in range(len(al)):
        for j in range(len(gm)):
            x_bar,t,h,e = func(gm=gm[j],al=al[i],*args,**kwargs)
            X[:,i,j] = x_bar[:,-1]
    return X,al,gm

def map_d_gamma(func,*args,**kwargs):
    
    d=np.arange(0.25,1.5,0.01)
    gm=np.arange(-6.0,6,0.1)
    X = np.zeros((N,len(d),len(gm)))
    for i in range(len(d)):
        for j in range(len(gm)):
            x_bar,t,h,e = func(gm=gm[j],d=d[i],*args,**kwargs)
            X[:,i,j] = x_bar[:,-1]
    return X,d,gm

def map_u_gamma(func,*args,**kwargs):
    
    u = np.arange(0,0.5,0.01)
    gm=np.arange(-6.0,6,0.1)
    X = np.zeros((N,len(u),len(gm)))
    for i in range(len(u)):
        for j in range(len(gm)):
            x_bar,t,h,e = func(gm=gm[j],u=u[i],*args,**kwargs)
            X[:,i,j] = x_bar[:,-1]
    return X,u,gm

def map_u_d(func,*args,**kwargs):
    
    u = np.arange(0,0.5,0.1)
    d=np.arange(0.25,1.5,0.1)
    X = np.zeros((N,len(u),len(d)))
    for i in range(len(u)):
        for j in range(len(d)):
            x_bar,t,h,e = func(d=d[j],u=u[i],*args,**kwargs)
            X[:,i,j] = x_bar[:,-1]
    return X,u,d

def plot_map(f,X,a,b,A):
    '''
    Plots the paramter map in a two dimensional parameter space for any two 
    paramters a and b

    Parameters
    ----------
    f : function
        function to chack if there is opinion cascade
    X : numpy array 3D
        consists of the solutions of the system for two varying parameters
    a : numpy array
        different values of paramter a
    b : numpy array
        different values of paramter b
    
    Returns
    -------
    None.

    '''
    
    cas=np.zeros((len(a),len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            cas[i,j] = f(A,X[:,i,j])
    bb,aa = np.meshgrid(b,a)
    bb=bb.flatten()
    aa=aa.flatten()
    cas = cas.flatten()
    # print(aa)
    # print(bb)
    # print(cas)
    plt.scatter(bb,aa,s=1,c=cas,cmap=plt.get_cmap('coolwarm'),vmin=0,vmax=1)
    plt.title("Map u gamma Path")
    plt.xlabel("gamma")
    plt.ylabel("u")
    plt.colorbar()
    plt.show()

# N = 40
# A=dy.sym_tree(3,3)
# #x0=(np.random.uniform(-1.0,1.0,size=N))
# x0=np.zeros(N-1)
# x0 = np.insert(x0,0,1.)
    
N=5
A=dy.path(N)
x0=(np.random.uniform(-1.0,1.0,size=N))
#x0 = np.array([0,0,1,0,0])
    

#X,al,gm = map_alpha_gamma(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,u=0.26,b=0.0)
#X,d,gm = map_d_gamma(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,u=0.26,al=1.2,b=0.0)
X,u,gm = map_u_gamma(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,al=1.2,b=0.0)
#X,u,d = map_u_d(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,gm=-1.3,al=1.2,b=0.0)

plot_map(cascade,X,u,gm,A)





