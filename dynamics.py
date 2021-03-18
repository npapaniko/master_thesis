# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:59:59 2021

@author: Nikos
"""

import numpy as np
import random
import sys
from iteration_utilities import unique_everseen
from random import randrange
import copy
import statistics
import matplotlib.pyplot as plt
from collections.abc import Iterable
import numpy.ma as ma
from itertools import zip_longest
from scipy.optimize import curve_fit



from tools import minority, assign_hyper_opinions, count_opinions, remove_dublicate
    
def influence(id_edge,hypergraph, opinions_hyper,op_dict): 
    """"Implements the influence step.
    Updates the opinions array and the opinions edge on the minority vertices"""
    op_hyperedge=opinions_hyper[id_edge]
    hyperedge=hypergraph[id_edge]
    minor=minority(op_hyperedge) #value of the minority opinion
    prob=op_hyperedge.count(minor)/len(op_hyperedge) #probability staying the same
    indices = [i for i, x in enumerate(op_hyperedge) if x == minor] #which nodes are in the minority
    # print(indices)
    # print(id_edge)
    # print(prob)
    for i in range(len(indices)):
        if random.random()<=1-prob: #with probability 1-p, we change the opinion
            node=hyperedge[indices[i]] #ID of the node with minority opinion
            op_dict[node]=op_dict[node]*(-1)+1
            opinions_hyper=assign_hyper_opinions(op_dict,hypergraph)
    return opinions_hyper, op_dict


def merge(hypergraph,opinions_hyper, op_dict, gamma):
    """"Calculates the hypergraph after splitted edges have merged with random edges
    The random edges must have a "tolerated" majority opinion equal with the opinion of the split edge
    Output: The new hypergraph with its opinion depiction"""
    aux = copy.deepcopy(hypergraph) #makes a copy of hypergraph
    aux.remove(hypergraph[-1]) #removes the last 2 edges which were split
    aux.remove(hypergraph[-2])
    random.shuffle(aux) #so that we choose randomly an edge to attach the split edge
    aux.sort(key=len) #so that we start attaching firstly the ones with smallest size
    op_aux=assign_hyper_opinions(op_dict, aux)
    edges_temp=[hypergraph[-1],hypergraph[-2]] #split edges array
    edges=copy.deepcopy(edges_temp) 
    op_edges=assign_hyper_opinions(op_dict,edges)
    
    hypergraph.remove(edges[0]) #removes the last 2 edges which were split from the final hypergraph
    hypergraph.remove(edges[1])
    

    for i in range(2): #going through the split edges of hypergraph
        
        for j in (range(len(aux))): #going through the small edges of aux
                
        
            edge_fr=count_opinions(op_aux[j]) #counts the fraction of 1s on each edge of aux
            if edge_fr==gamma: #if fraction is exactly on the borderline then nothing happens to avoid bias
                break
            elif (edge_fr< gamma) and op_edges[i][0]==0: #if "tolerated" majority of edge is 0 and split edge = 0

                hypergraph.remove(aux[j]) #removes the chosen random edge
                
                # if len(hypergraph)==0: #if the resulted hypergraph has no edges then break
                #     print("yes!!!!!!!!")
                #     break
                    
                
                    
                edges[i]=aux[j]+edges[i] #merge the random edge with split edge
                
                break #once we merge two edges we stop the loop
            elif edge_fr>1-gamma and op_edges[i][0]==1: #similarly for majority = 1 
                hypergraph.remove(aux[j]) #removes the chosen random edge
                
                # if len(hypergraph)==0:
                #     print("yes!!!!!!!!")

                #     break
                    
                    
                edges[i]=aux[j]+edges[i] #merge the random edge with split edge
               
                break
                
    edges[0]=remove_dublicate(edges[0]) #remove double nodes from the edge
    edges[1]=remove_dublicate(edges[1]) #remove double nodes from the edge
    # print("edgeeeeeeeeees")
    # print(edges)
    hypergraph.append(edges[0]) #attach new merged edge to hypergraph
    hypergraph.append(edges[1]) #attach new merged edge to hypergraph
    hypergraph=list(unique_everseen(hypergraph, key=frozenset)) #similar to remove_dublicate but for nested lists
    opinions_hyper=assign_hyper_opinions(op_dict, hypergraph) 

    return hypergraph, opinions_hyper


def split(id_edge, hypergraph,opinions_hyper,op_dict):
    """Splits a hyperedge into two hyperedges with the same kind of opinions"""
    
    hyperedge=hypergraph[id_edge] 
    splt_edges=[[num for num in hyperedge if op_dict[num]==o] for o in [0,1]] #splits the ID edge into 2 edges with same opinions
    hypergraph=[num for num in hypergraph if num!=hyperedge] #deletes the initial edge from hypergraph
    hyperedge=splt_edges #updates the split edge
    hypergraph.extend(hyperedge) #adds the new edges in hypergraph
    opinions_hyper=assign_hyper_opinions(op_dict,hypergraph) #creates an opinion depiction of the new hypergraph
    
    # print(hypergraph)
    # print(opinions_hyper)
    #merge mode: ON, strict majority : when 0.5
    hypergraph, opinions_hyper=merge(hypergraph,opinions_hyper, op_dict, 0.5)
    # print("Merging...")
    # print(hypergraph)
    # print(opinions_hyper)
    return hypergraph,opinions_hyper

            
        

def decide(id_edge,hypergraph,opinions_hyper,op_dict,gamma):
    """"Decides whether influence or splitting by calculating fraction of opinions 1 on hyperedge"""
    op_edge=opinions_hyper[id_edge] #choosing edge
    edge_fr=count_opinions(op_edge) #counting fraction of 1 
    # print("fraction of 1s is = {:5f}" .format(edge_fr))
    if gamma>0.5:
        gamma_cor=1-gamma
    else:
        gamma_cor=gamma
    if edge_fr<= gamma_cor or edge_fr>=1-gamma_cor:
        # print("Influencing. . .")
        opinions_hyper, op_dict=influence(id_edge,hypergraph, opinions_hyper,op_dict)
        # print(opinions_hyper)
    else: # gamma<=fraction<=1-gamma => splitting regime
        # print("Splitting. . .")
        hypergraph,opinions_hyper=split(id_edge, hypergraph,opinions_hyper,op_dict)
    return hypergraph, opinions_hyper,op_dict