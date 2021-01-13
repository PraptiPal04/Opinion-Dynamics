# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:43:50 2021

@author: Prapti
"""
import numpy as np

def rhs(x,A,d,u,al,gm,b):
    '''
    TO calculate the RHS of the given ode

    Parameters
    ----------
    x : numpy array
        curent state of the system
    A : numpy array
        the adjacency matrix of the system graph
    d : float
        parameter, resistance to becoming opinionated
    u : float
        control parameter, social influence
    al : float
        parameter, self reinforcement
    gm : float
        parameter, cooperative/competitive
    b : float
        parameter, input bias

    Returns
    -------
    x_dot : numpy array
        RHS of the ode. The derivative of the current state.

    '''
    x_dot=-d*x+u*np.tanh(al*x + gm * np.matmul(A,x))+b
    return x_dot


#TOPOLOGIES

def circle(N):
    '''
    Generates the adjacency matrix for a circular topology graph without self loops

    Parameters
    ----------
    N : integer
        number of nodes in the graph

    Returns
    -------
    A : numpy array of shape NxN
        generated adjacency matrix

    '''
    A=np.zeros((N,N))
    for i in range(N-1):
        A[i,i+1]=1
        A[i+1,i]=1
    
    A[N-1,0]=1
    A[0,N-1]=1
    return A
    
def wheel(N):
    '''
    Generates the adjacency matrix for a wheel topology graph without self loops

    Parameters
    ----------
    N : integer
        number of nodes in the graph

    Returns
    -------
    A : numpy array of shape NxN
        generated adjacency matrix

    '''
    A=np.zeros((N,N))
    for i in range(N-2):
        A[i,i+1]=1
        A[i+1,i]=1
        A[i,N-1]=1
        A[N-1,i]=1
    A[N-2,N-1]=1
    A[N-1,N-2]=1
    A[N-2,0]=1
    A[0,N-2]=1
    return A

def path(N):
    '''
    Generates the adjacency matrix for a path topology graph without self loops

    Parameters
    ----------
    N : integer
        number of nodes in the graph

    Returns
    -------
    A : numpy array of shape NxN
        generated adjacency matrix

    '''
    A=np.zeros((N,N))
    for i in range(N-1):
        A[i,i+1]=1
        A[i+1,i]=1
    return A

def star(N):
    '''
    Generates the adjacency matrix for a star topology graph without self loops

    Parameters
    ----------
    N : integer
        number of nodes in the graph

    Returns
    -------
    A : numpy array of shape NxN
        generated adjacency matrix

    '''
    A=np.zeros((N,N))
    for i in range(N-1):
        A[i,N-1]=1
        A[N-1,i]=1
    return A


