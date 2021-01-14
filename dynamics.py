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

#INFLUENCER
    
def add_followers(A, i, n):
    #adds n followers (star topology) of agent i to adj. matrix A and returns new adj. matrix
    old_n = A.shape[1]
    A = enlarge_matrix_N(A, n)
    for j in range(n):
        make_connection(A, i, old_n+j)
    return A

def influencer_network(N, m):
    #create network of n connected agents with m followers,
    #where m is of type int (adding m followers to each agent) or list of len(N) of ints (adding m[i] followers to agent i)
    A = mesh(N)
    for i in range(N):
        if isinstance(m, int):
            A = add_followers(A, i, m)
        else:
            A = add_followers(A, i, m[i])            
    return A

def enlarge_matrix(A):
    #takes matrix of any shape as input and returns adj. matrix with one row and one columns of zeros added
    B = np.ones([0,A.shape[1]+1])
    for row in A:
        B = np.append(B, [np.append(row, 0)], 0)
    B = np.append(B, [np.zeros(A.shape[1]+1)], 0)
    return B

def enlarge_matrix_N(A, n):
    #enlarge given matrix by n rows and columns of zeros
    for i in range(n):
        A = enlarge_matrix(A)
    return A

def make_connection(A, id1, id2):
    A[id1][id2] = 1
    A[id2][id1] = 1

def delete_connection(A, id1, id2):
    A[id1][id2] = 0
    A[id2][id1] = 0


def mesh(N):
    A = np.ones((N, N))
    for i in range(N):
        A[i, i] = 0
    return A