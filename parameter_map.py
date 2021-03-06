# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:43:43 2021

@author: Prapti
"""


import numpy as np
import matplotlib.pyplot as plt
from RKF import rkf
import dynamics as dy
import graph_plot as gr


#Mapping attention parameter u

def map_u(func,*args,**kwargs):
    
    u = np.arange(0,0.5,0.01)
    X = np.zeros((N,len(u)))
    for i in range(len(u)):
        x_bar,t,h,e = func(u=u[i],*args,**kwargs)
        X[:,i] = x_bar[:,-1]
    # for i in range(N):
    #     plt.plot(u,X[i,:])
    return X,u
    
#Mapping resistence term d
        
def map_d(func,*args,**kwargs):
    
    d=np.arange(0.25,1.5,0.01)
    X = np.zeros((N,len(d)))
    for i in range(len(d)):
        x_bar,t,h,e = func(d=d[i],*args,**kwargs)
        X[:,i] = x_bar[:,-1]
    
    # for i in range(N):
    #     plt.plot(d,X[i,:])
    return X,d


#Mapping self reinforcement alpha
        
def map_al(func,*args,**kwargs):
    
    al=np.arange(-1.0,8,0.05)
    X = np.zeros((N,len(al)))
    for i in range(len(al)):
        x_bar,t,h,e = func(al=al[i],*args,**kwargs)
        X[:,i] = x_bar[:,-1]
    
    for i in range(N):
        plt.plot(al,X[i,:])
    return X,al



#Mapping gamma
        
def map_gm(func,*args,**kwargs):
    
    gm=np.arange(-6.0,6,0.05)
    X = np.zeros((N,len(gm)))
    for i in range(len(gm)):
        x_bar,t,h,e = func(gm=gm[i],*args,**kwargs)
        X[:,i] = x_bar[:,-1]
    
    # for i in range(N):
    #     plt.plot(gm,X[i,:])
    return X,gm


# #Mapping external bias b
        
def map_b(func,*args,**kwargs):
    
    b=np.arange(-0.3,0.3,0.05)
    X = np.zeros((N,len(b)))
    for i in range(len(b)):
        x_bar,t,h,e = func(b=b[i],*args,**kwargs)
        X[:,i] = x_bar[:,-1]
    
    for i in range(N):
        plt.plot(b,X[i,:])


N=5
A=dy.path(N)
x0=(np.random.uniform(-1.0,1.0,size=N))
#x0=np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

#x,u=map_u(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,al=1.2,gm=1.3,b=0.0)
#x,d=map_d(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,u=0.26,al=1.2,gm=-1.3,b=0.0)
x,al=map_al(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,u=0.26,gm=-1.3,b=0.0)
#x,gm=map_gm(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,u=0.26,al=1.2,b=0.0)
#x=map_b(rkf,f=dy.rhs,x=x0,h=0.05,N=N,A=A,d=0.5,u=0.26,al=1.3,gm=-1.3)


    
# plt.title("Parameter Map : gamma (alpha = 4.0,d=1.0) Path Topology")
# plt.ylabel("Opinions x")
# plt.xlabel("Parameter gamma")
# #plt.savefig("u_map_Path_Agreement.jpg")
plt.show()


#gr.graph(A,x)




    