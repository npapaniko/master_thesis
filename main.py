# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:40:14 2021

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



from iterations import statistics_calc_multi
from hypergraph_set_up import create_hypergraph
from tools import assign_hyper_opinions, assign_opinions

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
    
def figure_generator(num_sp_array,size_array,ratio_ones_array): 
    """"Plots the histograms of the multiple simulations"""
    #bins = range(10)
    
    fig, ax = plt.subplots(1, 3)


    ax[0].hist(num_sp_array, bins=np.arange(min(num_sp_array), max(num_sp_array) + 0.5, 0.5))
    ax[0].set(xlabel="Number of edges",ylabel="Counts")
    # bins_labels(bins, fontsize=7)
    
    ax[1].hist(size_array,bins=np.arange(min(size_array), max(size_array) + 0.5, 0.5))
    ax[1].set(xlabel="Size of edges",ylabel="Counts")

    # bins_labels(bins,fontsize=7)
    
    ax[2].hist(ratio_ones_array,bins=np.arange(min(ratio_ones_array), max(ratio_ones_array) + 0.005, 0.005))
    ax[2].set(xlabel="Ratio of opinion 1 over total number of network",ylabel="Counts")

    # bins_labels(bins,fontsize=7)
    
    plt.show()
    
def varying_parameters(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals):
    """For many values of a parameter, calculates multiple simulations from an initial hypergraph
        Output: DISTRIBUTIONS for multiple simulations"""
    fig, ax=plt.subplots(num_intervals,3)
    gamma=np.linspace(0.1,0.49,num_intervals)
    for i in range(num_intervals):
        
        num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array=statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma[i])
        
        ax[i,0].hist(num_sp_array, bins=np.arange(min(num_sp_array), max(num_sp_array) + 0.5, 0.5))
        ax[i,0].set(xlabel="Number of edges",ylabel="Counts")
        
        ax[i,1].hist(size_array,bins=np.arange(min(size_array), max(size_array) + 0.5, 0.5))   
        ax[i,1].set(xlabel="Size of edge",ylabel="Counts")
        
        ax[i,2].hist(ratio_ones_array,bins=np.arange(min(ratio_ones_array), max(ratio_ones_array) + 0.005, 0.005))
        ax[i,2].set(xlabel="Ratio of opinion 1 over total number of network",ylabel="Counts")
    
    plt.tight_layout()
    plt.show()    
    return 

def magnetization_vs_density(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals):
    """Plots magnetization versus density for a given initial network and for different values of gamma """
    gamma=np.linspace(0.1,0.49,num_intervals)
    rho_simulations=[0]*num_intervals
    magnetization_simulations=[0]*num_intervals
    # n_edges_simulations=[0]*num_intervals
    # num_edges_average=[0]*num_intervals
    # rho_simulations_average=[0]*num_intervals
    cmap = plt.cm.get_cmap('hsv', num_intervals+1)
    
    for i in range(num_intervals):
        num_sp_array,size_array,ratio_ones_array, magnetization_simulations[i], rho_simulations[i]=statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma[i])
        # num_edges_average[i]=[np.ma.average(ma.masked_values(temp_list, None)) for temp_list in zip_longest(*n_edges_simulations[i])]
        

        #rho_simulations_average[i]=[np.ma.average(ma.masked_values(temp_list, None)) for temp_list in zip_longest(*rho_simulations[i])]
        
        # print(len(n_edges_simulations))
        # print(len(rho_simulations))
        # print(len(n_edges_simulations[i]))
        # print(len(rho_simulations[i]))
        
        #num_sp_simulations[i]=sum(num_sp_array)/len(num_sp_array)
        
        for j in range(num_simulations):
            
                plt.scatter(rho_simulations[i][j],magnetization_simulations[i][j],s=2,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "")
                # plt.scatter(rho_simulations[i][j],n_edges_simulations[i][j],s=10,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "")
        # plt.scatter(rho_simulations_average[i],num_edges_average[i],s=2,color=cmap(i+num_intervals),label="γ= %.2f" %gamma[i] if i == 0 else "")

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("Density of inactive edges")
    plt.ylabel("Magnetization")
    # plt.ylabel("Number of edges")

    plt.show()
    
    return


def flatten(lis):
    """"Converts a list of lists into a list e.g. [3,[4,5]] -> [3,4,5]"""
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item

def function_fit(x, a,c):
	 return (a*(x))**(c)
    # return (a*x)**(1/c)

def fit_m_vs_density(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, gamma):
    """Calculates coefficients of a fitted parabola and plots it on the m vs rho diagram"""
    num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array =statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma)
    y_coords=list(flatten(m_multi_array))
    x_coords=list(flatten(rho_multi_array))
    points=list(zip(x_coords, y_coords))
    sorted(points , key=lambda k: [k[1], k[0]])
    points=np.array(points)
    y_coords_fit=points[:,1]
    x_coords_fit=points[:,0]
    
    x_ave_fit, y_ave_fit = average_m_vs_density(num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array)
    
    # params =  curve_fit(lambda x, a, b,c: function_fit(x, gamma, a, b,c), [1,1,1], y_ave_fit , x_ave_fit )
    params =  curve_fit(function_fit, y_ave_fit , x_ave_fit,[1,4] )

    a,c=params[0]
    
    y=np.linspace(min(y_coords_fit),max(y_coords_fit),1000)
    x=function_fit(y,a,c)

    
    print("The coefficients of the parabola are:a={}, c={}" .format(a,c))
    plt.scatter(x_coords,y_coords,s=2,color="orange")
    plt.scatter(x_ave_fit, y_ave_fit,s=2,color="black")
    plt.plot(x, y,color="blue")
    
    # plt.scatter(y_coords,x_coords,s=2,color="orange")
    # plt.scatter(y_ave_fit, x_ave_fit,s=2,color="black")
    # plt.plot(y, x,color="blue")
    plt.show()
    return

def average_m_vs_density(num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array):
    y_coords=list(flatten(m_multi_array))
    x_coords=list(flatten(rho_multi_array))
    points=list(zip(x_coords, y_coords))
    sorted(points , key=lambda k: [k[1], k[0]])
    points=np.array(points)
    # print(points)
    # print("--------")
    points_positive=[[x,y] for x,y in points if 0 <= y]
    points_negative=[[x,y] for x,y in points if 0 >= y]
    points_truncated=[]
    
    if len(points_positive)>len(points_negative):
        points_truncated=points_positive
    else:
        points_truncated=points_negative
    
    # print(points_pos)
    y_coords_fit=[item[1] for item in points_truncated]
    x_coords_fit=[item[0] for item in points_truncated]
    

    dt=0.001
    init=0
    final=1
    intervals=np.arange(init,final,dt)
    
    category_m_xcoord=[[] for _ in range(len(intervals))]
    category_m_ycoord=[[] for _ in range(len(intervals))]
    average_m_xcoord=[[] for _ in range(len(intervals))]
    average_m_ycoord=[[] for _ in range(len(intervals))]
    intervals=list(intervals)
    intervals_copied=copy.deepcopy(intervals)

    for i in range(len(intervals)):
        for j in range(len(x_coords_fit)):
            # print(x_coords_fit[j])
            if x_coords_fit[j]>=intervals[i] and x_coords_fit[j]<=intervals[i]+dt:
                # print("yes")
                # print(x_coords_fit[j])
                # print(category_m_xcoord)
                category_m_xcoord[i].append(x_coords_fit[j])
                category_m_ycoord[i].append(y_coords_fit[j])
                # print(interval)
                # print("-------")
                # print(category_m_xcoord[i])
                # print("-------")
        # print(category_m_xcoord[i])
        # print(i)
        if len(category_m_ycoord[i])!=0:
            average_m_xcoord[i]=sum(category_m_xcoord[i])/len(category_m_xcoord[i])
            average_m_ycoord[i]=sum(category_m_ycoord[i])/len(category_m_ycoord[i])
        else:
            # average_m_xcoord.remove(average_m_xcoord[i])
            intervals_copied.remove(intervals[i])
            
            

    average_m_ycoord = [e for e in average_m_ycoord if e]
    average_m_xcoord = [e for e in average_m_xcoord if isinstance(e, (float, int))]

    plt.scatter(x_coords,y_coords,s=6,color="orange")
    plt.scatter(average_m_xcoord,average_m_ycoord,s=3,color="black")
    plt.show()
    
    return average_m_xcoord, average_m_ycoord
    
    


    
#------------------------Parameters------------------------------#
N=300 #number of nodes
S=50 #size of hyperedges
n=20 #number of hyperedges
gamma=0.49 #threshold for split
num_it=1000 #number of iterations 
num_simulations=50 #number of simulations for statistics
num_intervals=1 #number of different gamma values for magnetization vs density function
if S*n<N:
    print("S*n<N: Error: Every node must be in at least one hyperedge of size S")
    sys.exit()
    
#----------------------------------------------------------------#



H=create_hypergraph(N,S,n)
op_dict=assign_opinions(N)
opinions_H=assign_hyper_opinions(op_dict,H)

print(op_dict)
print("----------")
print(H)
print("----------")
print(opinions_H)
print("----------")

# H,opinions_H,op_dict,m_array, rho_array=iterations(H,op_dict,num_it,gamma)
# print(H)
# print(opinions_H)
# print(m_array)
# print(rho_array)

# varying_parameters(num_simulations,H, opinions_H, op_dict,num_it, num_intervals)

num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array=statistics_calc_multi(num_simulations,H, opinions_H, op_dict,num_it,gamma)

# figure_generator(num_sp_array,size_array,ratio_ones_array)

# magnetization_vs_density(num_simulations,H, opinions_H, op_dict,num_it, num_intervals)
average_m_vs_density(num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array)


# fit_m_vs_density(num_simulations,H, opinions_H, op_dict,num_it,gamma)

