# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:04:52 2021

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
    

from tools import assign_hyper_opinions, magnetization_density, count_opinions
from dynamics import decide

def iterations(hypergraph,op_dict,num_it,gamma):
    """Iterates decide for num_it times with input as output at every step
    The m_array and rho_array have the magnetization of the whole hypergraph for each iteration"""
    opinions_hypergraph=assign_hyper_opinions(op_dict,hypergraph)
    m_array=[]
    rho_array=[]
    # n_edges=[]
    for i in range(num_it):
        id_edge=randrange(len(hypergraph)) #chooses a random hyperedge of the hypergraph
        
        m, rho= magnetization_density(opinions_hypergraph)
        m_array.append(m)
        rho_array.append(rho)
        # n_edges.append(len(hypergraph))
        hypergraph, opinions_hypergraph,op_dict=decide(id_edge,hypergraph,opinions_hypergraph,op_dict,gamma) #split or influence
        
        # print(hypergraph)
        # print(opinions_hypergraph)
        # print("--")
        # print(opinions_hypergraph)

        # print(m)
        # print(rho)

        if all([len(set(num)) == 1 for num in opinions_hypergraph]):
            print("Reached Equilibrium at iteration {:d}" .format(i))
            break
        
    return hypergraph,opinions_hypergraph,op_dict, m_array, rho_array
        

def statistics_calc(hypergraph, opinions_hypergraph, op_dict,num_it,gamma):
    """"Calculates for the final iterated hypergraph the following:
        num_species = number of edges
        size_speces = array with the size of each edge
        ratio_ones = the fraction of nodes with ones in the whole hypergraph
        m_array = same- unchanged m_array with iterations function
        rho_array = same - unchanged rho_array with iterations function"""
    hypergraph,opinions_hypergraph,op_dict, m_array, rho_array =iterations(hypergraph,op_dict,num_it,gamma) 
    
    num_species=len(hypergraph) 
    
    size_species=np.zeros(len(hypergraph))
    ratio_ones_each=np.zeros(len(hypergraph)) 
    ratio_ones=0
    # print(hypergraph)
    # print(opinions_hypergraph)
    for i in range(len(hypergraph)):
        size_species[i]=len(hypergraph[i])
        ratio_ones_each[i]=count_opinions(opinions_hypergraph[i])
    numerator= [a*b for a,b in zip(size_species,ratio_ones_each)]
    ratio_ones= sum(numerator)/sum(size_species) #fraction of whole ones = (l1*f1+l2*f2+...)/(l1+l2+...) l= size of edge, l*f= number of ones
    return num_species, size_species, ratio_ones, m_array, rho_array

def statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma):
    """For an initial hypergraph, it calculates arrays up to :
        num_species = number of edges for each simulation
        size_speces = array with the size of each edge for each simulation
        ratio_ones = the fraction of nodes with ones in the whole hypergraph for each simulation 
        m_multi_array = list of m_array for every single simulation
        rho_multi_array = list of rho_array for every single simulation """
    size_array=[]
    m_multi_array=[]
    rho_multi_array=[]
    num_sp_array=np.zeros(num_simulations)
    ratio_ones_array=np.zeros(num_simulations)
    
    # n_edges_array=[]
    for i in range(num_simulations):

        hypergraph_init=copy.deepcopy(hypergraph) #after each simulation, we want the initial hypergraph to be the given one
        opinions_init=copy.deepcopy(opinions_hypergraph)
        op_dict_init=copy.deepcopy(op_dict)

        num_sp_array[i], size_var, ratio_ones_array[i], m_array, rho_array =statistics_calc(hypergraph_init, opinions_init, op_dict_init,num_it,gamma)
        
        # n_edges_array.append(n_edges)
        m_multi_array.append(m_array)
        rho_multi_array.append(rho_array)
        
        size_array.extend(size_var)
        # print(rho_multi_array)
        # print("-----------")        
    return num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array
