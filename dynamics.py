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

#Influencer Network
def add_followers(A, i, n):
    """
    adds n followers (star topology) of agent i to adj. matrix A and 
    returns new adj. matrix
    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix
    i : integer - ID of agent in network that followers shall be added to
    n : integer - number of followers to be added
    Returns
    -------
    A :  np.array (matrix) - adjacency matrix
    """
    old_n = A.shape[1]
    A = enlarge_matrix_N(A, n)
    for j in range(n):
        A = make_connection(A, i, old_n+j)
    return A

def influencer_network(N, m):
    """
    Create network of N connected agents with m followers, where m is of type 
    int (adding m followers to each agent) or list of len(N) of ints (adding 
    m[i] followers to agent i)
    Parameters
    ----------
    N : Integer - Number of influencers
    m : Integer or List of Integers - Number of followers for each influencer
    Returns
    -------
    A : np.array (matrix) - adjacency matrix
    """
    A = mesh(N)
    for i in range(N):
        if isinstance(m, int):
            A = add_followers(A, i, m)
        else:
            A = add_followers(A, i, m[i])            
    return A



#Matrix transformation functions

def enlarge_matrix(A):
    """
    takes matrix of any shape as input and returns adj. matrix with one row 
    and one columns of zeros added
    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix
    Returns
    -------
    B : np.array (matrix) - enlarged adjacency matrix
    """
    B = np.zeros([A.shape[0]+1, A.shape[1]+1])
    B[:-1, :-1] = A
    return B

def enlarge_matrix_N(A, n):
    """
    enlarges given matrix by n rows and columns of zeros
    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix
    n : integer - number of rows and columns A will be enlarged by
    Returns
    -------
    A : np.array (matrix) - enlarged adjacency matrix
    """
    for i in range(n):
        A = enlarge_matrix(A)
    return A

def make_connection(A, id1, id2):
    """
    makes a connection betwween two agents in an adjacency matrix
    Parameters
    ----------
    A :   np.array (matrix) - input adjacency matrix
    id1 : integer - ID of one of the agents in A that is going to be connected
    id2 : integer - ID of the other agent in A that is going to be connected
    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A[id1][id2] = 1
    A[id2][id1] = 1
    return A

def delete_connection(A, id1, id2):
    
    """
    deletes a connection betwween two agents in an adjacency matrix
    Parameters
    ----------
    A :     np.array (matrix) - input adjacency matrix
    id1 :   integer - ID of one of the agents in A that is going to be 
            disconnected
    id2 :   integer - ID of the other agent in A that is going to be
            disconnected
    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A[id1][id2] = 0
    A[id2][id1] = 0
    return A



# Random Tree Topology
def add_rand_branches(A, low, high):
    """
    Adds a (random) number of branches to an existing tree. New branches will 
    only be added to a node, if that node is only connected with one other 
    node (so if it is on the outside of the tree). The random number can be 
    bounded and therefore also fixed on a not random int by setting low=high
    Parameters
    ----------
    A :   Adjacency matrix of network that the branches shall be added to
    low : lowest possible integer for the random choice of added branches to 
          each node
    high :highest possible integer for the random choice of added branches to 
          each node 
          
          NOTE: if low = high = n, then n branches will be added to each node
    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A_copy = A
    counter = 0
    for row in A_copy:
        if np.sum(row) < 2:
            n = np.random.randint(low=low, high=high+1)
            A = add_followers(A, counter, n)
        counter += 1
    return A

def rand_tree(depth, low, high):
    """
    Creates a tree topology with the "depth" being the number of nodes 
    counting from the central node and going the longest path to end of tree
    (length of all paths is identical if low >0)
    Parameters
    ----------
    depth : integer, depth as defined above 
    low :   integer "low" that is passed to add_rand_branches function (see
            docstring there)
    high :  integer "high" that is passed to add_rand_branches function (see
            docstring there)
    Returns
    -------
    A : np.array (matrix) - adjacency matrix
    """
    A = np.zeros([1, 1])
    for _ in range(depth):
        A = add_rand_branches(A, low, high)
    return A       


# Symmetric Tree Topology
def add_sym_branches(A, n):
    """
    Adds a n number of branches toeach node in an existing tree. New branches will 
    only be added to a node, if that node is only connected with one other 
    node (so if it is on the outside of the tree). 
    Parameters
    ----------
    A :   Adjacency matrix of network that the branches shall be added to
    n : integer, number of nodes added to each node
    
    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A_copy = A
    counter = 0
    for row in A_copy:
        if np.sum(row) < 2:
            #n = np.random.randint(low=low, high=high+1)
            A = add_followers(A, counter, n)
        counter += 1
    return A

def sym_tree(lvl,n):
    """
    Creates a tree topology with the lvl being the number of nodes 
    counting after the central node and going the longest path to end of tree
    At each level there are n nodes connected to each node
    
    Parameters
    ----------
    lvl : integer, level as defined above 
    n :   integer, number of nodes connected to each node at any particular 
    level
    
    Returns
    -------
    A : np.array (matrix) - adjacency matrix
    """
    A = np.zeros([1, 1])
    for _ in range(lvl):
        A = add_sym_branches(A, n)
    return A


