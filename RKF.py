# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:41:20 2021

@author: Prapti
"""
import numpy as np
# import dynamics as dy
# import matplotlib.pyplot as plt
# import graph_plot as gr


#Butcher Tableau
alpha = np.array([0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0])
beta = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
		[1932.0/2197.0, (-7200.0)/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
		[439.0/216.0, -8.0, 3680.0/513.0, (-845.0)/4104.0, 0.0, 0.0],
		[(-8.0)/27.0, 2.0, (-3544.0)/2565.0, 1859.0/4104.0, (-11.0)/40.0, 0.0]])
c = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, (-1.0)/5.0, 0.0]) # coefficients for 4th order method
c_star = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, (-9.0)/50.0, 2.0/55.0]) # coefficients for 5th order method
cerr=c-c_star

tol = 0.0000001     #desired accuracy/tolerance
safe = 0.84     #safety factor
N_t = 200  #no. of steps taken in the adaptive step size
h_max = 3.

def rkf(f,x,h,N,*args,**kwargs):
    '''
    To perform integration stepwise using the Runge-Kutta-Fehlberg 4(5) method with adaptive step size control
    Parameters
    ----------
    f : function
        calculating the rhs of the ode
    x : numpy array
        initial state of the system
    h : float
        initial stepsize
    alpha : numpy array
        alpha values of the RK-4(5) method
    beta : numpy array
        beta values of the RK-4(5) method
    cerr : numpy array
        difference in the c cooefficients of the RK-4 and RK-5 methods
    c_star : numpy array
        c values of the RK-5 method
    *args : additional arguments to passdown to function f
    **kwargs : additional keywork arguments to pass down to function f
    Returns
    -------
    X : numpy array
        calculated values of the system for each time step
    T : numpy array
        time values corresponding to each time step in the system
    H : numpy array
        adaptive step size values for every calculated step
    EPS : numpy array
        error epsilon at each step(between the calculated values from RK-4 and RK-5 method)
    '''
    T = np.zeros((1))
    X = np.zeros((N,1))
    H = np.array([h])
    EPS = np.zeros((1))
    X[:,0] = x
    
    i = 0
    while i<N_t: 
        k1 = f(X[:,i],*args,**kwargs)
        k2 = f(X[:,i]+h*beta[1,0]*k1, *args,**kwargs)
        k3 = f(X[:,i]+h*(beta[2,0]*k1 + beta[2,1]*k2),*args,**kwargs)
        k4 = f(X[:,i]+h*(beta[3,0]*k1 + beta[3,1]*k2 + beta[3,2]*k3),*args,**kwargs)
        k5 = f(X[:,i]+h*(beta[4,0]*k1 + beta[4,1]*k2 + beta[4,2]*k3 + beta[4,3]*k4),*args,**kwargs)
        k6 = f(X[:,i]+h*(beta[5,0]*k1 + beta[5,1]*k2 + beta[5,2]*k3 + beta[5,3]*k4 + beta[5,4]*k5),*args,**kwargs)
        errorfield = h*(cerr[0]*k1 + cerr[1]*k2 + cerr[2]*k3 + cerr[3]*k4 + cerr[4]*k5 + cerr[5]*k6)
        max_error = np.absolute(errorfield).max()
        
        if (max_error <= tol):
            temp = X[:,i] + h*(c_star[0]*k1 + c_star[1]*k2 + c_star[2]*k3 + c_star[3]*k4 + c_star[4]*k5 + c_star[5]*k6)
            X = np.insert(X,i+1,temp,axis=1)
            h = safe * h* (tol/max_error)**0.2
            H=np.append(H,h)
            T=np.append(T,T[i]+h)
            EPS=np.append(EPS,max_error)
            i+=1
        else:
            h=safe*h*(tol/max_error)**0.25
    
    return X,T,H,EPS


def rkf_var(f,x,h,N,d_old,u_old,g,k,*args,**kwargs):
    '''
    To perform integration stepwise using the Runge-Kutta-Fehlberg 4(5) method with adaptive step size control

    Parameters
    ----------
    f : function
        calculating the rhs of the ode
    x : numpy array
        initial state of the system
    h : float
        initial stepsize
    alpha : numpy array
        alpha values of the RK-4(5) method
    beta : numpy array
        beta values of the RK-4(5) method
    cerr : numpy array
        difference in the c cooefficients of the RK-4 and RK-5 methods
    c_star : numpy array
        c values of the RK-5 method
    *args : additional arguments to passdown to function f
    **kwargs : additional keywork arguments to pass down to function f

    Returns
    -------
    X : numpy array
        calculated values of the system for each time step
    T : numpy array
        time values corresponding to each time step in the system
    H : numpy array
        adaptive step size values for every calculated step
    EPS : numpy array
        error epsilon at each step(between the calculated values from RK-4 and RK-5 method)

    '''
    T = np.zeros((1))
    X = np.zeros((N,1))
    H = np.array([h])
    EPS = np.zeros((1))
    X[:,0] = x
    U = np.array([u_old])
    D = np.array([d_old])
        
    
    i = 0
    while h<h_max : 
        
        U = np.append(U,k(U[i],T[i]))
        D = np.append(D,g(D[i],T[i]))
        
        k1 = f(X[:,i], d=D[i],u=U[i],*args,**kwargs)
        k2 = f(X[:,i]+h*beta[1,0]*k1, d=D[i],u=U[i], *args,**kwargs)
        k3 = f(X[:,i]+h*(beta[2,0]*k1 + beta[2,1]*k2), d=D[i],u=U[i],*args,**kwargs)
        k4 = f(X[:,i]+h*(beta[3,0]*k1 + beta[3,1]*k2 + beta[3,2]*k3), d=D[i],u=U[i],*args,**kwargs)
        k5 = f(X[:,i]+h*(beta[4,0]*k1 + beta[4,1]*k2 + beta[4,2]*k3 + beta[4,3]*k4), d=D[i],u=U[i],*args,**kwargs)
        k6 = f(X[:,i]+h*(beta[5,0]*k1 + beta[5,1]*k2 + beta[5,2]*k3 + beta[5,3]*k4 + beta[5,4]*k5),d=D[i],u=U[i],*args,**kwargs)
        errorfield = h*(cerr[0]*k1 + cerr[1]*k2 + cerr[2]*k3 + cerr[3]*k4 + cerr[4]*k5 + cerr[5]*k6)
        max_error = np.absolute(errorfield).max()
        
        if (max_error <= tol):
            temp = X[:,i] + h*(c_star[0]*k1 + c_star[1]*k2 + c_star[2]*k3 + c_star[3]*k4 + c_star[4]*k5 + c_star[5]*k6)
            X = np.insert(X,i+1,temp,axis=1)
            h = safe * h* (tol/max_error)**0.2
            H=np.append(H,h)
            T=np.append(T,T[i]+h)
            EPS=np.append(EPS,max_error)
            i+=1
        else:
            h=safe*h*(tol/max_error)**0.25
        
    
    return X,T,H,EPS,D,U

# A = dy.path(5)
# #x0 = (np.random.uniform(-1.0,1.0,size=5))
# x0=np.array([0,0,1,0,0])
# x,t,h,e = rkf(dy.rhs,x0,0.05,5,A=A,d=0.5,u=0.26,al=0,gm=-3,b=0.0)
# # # for i in range(5):
# # #     plt.plot(t,x[i,:],'.:')
# # plt.plot(t,h,'r.:')
# gr.graph_plot(A,x[:,-1])
# plt.show()

