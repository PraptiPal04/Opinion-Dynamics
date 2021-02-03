# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:32:18 2021

@author: Prapti
"""
import networkx as nx
import matplotlib as mpl
import matplotlib.animation as an
import numpy as np
import matplotlib.pyplot as plt

def graph_animate(A,x):
    '''
    Creates graph from the adjacency matrix and plots the network.
    An animation video is also created and saved of the evolution

    Parameters
    ----------
    A : numpy array
        adjacency matrix of the system
    x : numpy array
        solution of the system for every time step
   

    Returns
    -------
    None.

    '''
    rows,cols=np.where(A==1.)
    edges=zip(rows.tolist(),cols.tolist())
    G=nx.Graph()
    G.add_edges_from(edges)
   # plot_positions=nx.drawing.circular_layout(G)
    plot_positions = nx.circular_layout(G)
    new_dict = dict(zip(sorted(G.nodes()),plot_positions.values()))
    plot_positions = new_dict

    vmin=-1
    vmax=1
    norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap=plt.get_cmap('coolwarm')
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    labels={}
    for i in range(np.shape(A)[0]):
        labels[i] = f"{i}"
    
    def animate(i):
        '''
        Function to iterate over in order to animate the graphs

        Parameters
        ----------
        i : int
        interative variable for the FuncAnimation() function to animate the graph

        Returns
        -------
        None.

        '''
        nx.draw(G,pos=plot_positions,node_size=500,node_color=x[:,i],cmap='coolwarm',vmin=vmin,vmax=vmax)
        # string = s+str(v[i])
        #nx.draw_networkx_labels(G, plot_positions, labels)
        plt.title(" Connections : 2 ; Probability : 0.25 ; with bias")
    fig=plt.gcf()
    plt.colorbar(sm)
    anim = an.FuncAnimation(fig, animate, frames=200, blit=False)
    writervideo = an.FFMpegWriter(fps=10) 
    anim.save('Small_world_prob_bias_dyn_d_10.mp4', writer=writervideo)

def graph_plot(A,x):
    
    '''
    Creates graph from the adjacency matrix and plots the network.
    

    Parameters
    ----------
    A : numpy array
        adjacency matrix of the system
    x : numpy array
        solution of the system for every time step
   

    Returns
    -------
    None.

    '''
    rows,cols=np.where(A==1.)
    edges=zip(rows.tolist(),cols.tolist())
    G=nx.Graph()
    G.add_edges_from(edges)
    plot_positions=nx.drawing.spring_layout(G)

    vmin=-1
    vmax=1
    norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap=plt.get_cmap('coolwarm')
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    nx.draw(G,pos=plot_positions,node_size=500,node_color=x,cmap='coolwarm',vmin=vmin,vmax=vmax)
    plt.colorbar(sm)
    
    
    
    
    
    
    
    