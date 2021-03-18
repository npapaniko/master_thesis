# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:11:41 2021

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






def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def minority(op_hyperedge):
    """"Calculates which opinion is a minority in the edge. 
    Used for influence to apply the update rules on the minority edges"""
    i=0
    if op_hyperedge.count(1)>op_hyperedge.count(0):
        i=0
    else:
        i=1
    return i

def assign_opinions(N):
    """"Creates an array with random opinions. 
    Each index corresponds to the node of the network => Assigns an opinion to every node"""
    nodes=list(range(N))
    opinions=np.random.randint(2, size=N).tolist() #INSTEAD OF RANDOM DO DETERMINISTICALLY TO REDUCE NOISE
    op_dict=dict(zip(nodes,opinions))
    return op_dict

def assign_hyper_opinions(op_dict,hypergraph):
    """Depicts the hypergraph list into the opinions of the nodes instead of the ID of each node"""

    opinions_hyper=[]
    for i in range(len(hypergraph)):
        opinions_hyper.extend([[op_dict[key] for key in hypergraph[i]]]) #Creates superlist, with lists of the opinions of the nodes on each edge
    return opinions_hyper


def remove_dublicate(seq):
    """"Removes nodes that are twice in an edge"""
    seen = set() 
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def count_opinions(opinions_hyperedge):
    """"Counts the fraction of 1 opinions on the hyperedge"""
    edge_fr=opinions_hyperedge.count(1)/len(opinions_hyperedge)
    return edge_fr


def magnetization_density(opinions_hyper):
    """"Calculates the magnetization and the density of active edges of the hypergraph: 
        m= number_ones-number_zeros 
        rho = number_non_uniform_edges"""
    m=0
    rho=0
    for i in range(len(opinions_hyper)):
        m+=opinions_hyper[i].count(1)-opinions_hyper[i].count(0)
        if len(set(opinions_hyper[i])) == 1:
            rho+=1/len(opinions_hyper)
    return m, rho
