# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:55:48 2021

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



        
def create_hypergraph(N,S,n):
    """"Create a S-uniform (all edges size S) hypergraph of N nodes with n hyperedges
    --------------------------------------------------------------------------------------
        Output: a list of hyperedges
        First stage: Making sure *all* nodes are sorted into hyperedges.
        Second stage: If first stage is completed AND a hyperedge does not have size S
        then: random nodes are filled in the empty spaces
        Third stage: For the rest of the hyperedges, S random nodes are chosen repeatedly"""
    nodes=list(range(N))
    random.shuffle(nodes)
    hypergraph=[nodes[i:i + S] for i in range(0, len(nodes), S)] #Splits nodes list into lists with size S
    if len(hypergraph[-1])!=len(hypergraph[0]):  #if there are empty slots in the last hyperedge
    
        nodes_reduced = [x for x in nodes if x not in hypergraph[-1]] #making sure no repetition of nodes in hyperedge
        random.shuffle(nodes_reduced)
        hypergraph[-1].extend(nodes_reduced[:(len(hypergraph[0])-len(hypergraph[-1]))]) #fills the empty slots of the last hyperedge
        
    if np.ceil(N/S)<n: #For the remaining edges, we choose repeatedly S random nodes
        rmn=n-np.ceil(N/S) #number of remaining edges
        for i in range(int(rmn)):
            random.shuffle(nodes)
            hypergraph.extend([nodes[:S]])
            
    hypergraph=list(unique_everseen(hypergraph, key=frozenset))

    return hypergraph

