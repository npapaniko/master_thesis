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

from scipy.optimize import curve_fit


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
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
    return hypergraph

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


def minority(op_hyperedge):
    """"Calculates which opinion is a minority in the edge. 
    Used for influence to apply the update rules on the minority edges"""
    i=0
    if op_hyperedge.count(1)>op_hyperedge.count(0):
        i=0
    else:
        i=1
    return i
    
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

def remove_dublicate(seq):
    """"Removes nodes that are twice in an edge"""
    seen = set() 
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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
    
    # print("Splitting...")
    # print(hypergraph)
    # print(opinions_hyper)
    #merge mode: ON, strict majority : when 0.5
    hypergraph, opinions_hyper=merge(hypergraph,opinions_hyper, op_dict, 0.5)
    # print("Merging...")
    # print(hypergraph)
    # print(opinions_hyper)
    return hypergraph,opinions_hyper
                    

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
    
def iterations(hypergraph,op_dict,num_it,gamma):
    """Iterates decide for num_it times with input as output at every step
    The m_array and rho_array have the magnetization of the whole hypergraph for each iteration"""
    opinions_hypergraph=assign_hyper_opinions(op_dict,hypergraph)
    m_array=[]
    rho_array=[]
    for i in range(num_it):
        id_edge=randrange(len(hypergraph)) #chooses a random hyperedge of the hypergraph
        
        m, rho= magnetization_density(opinions_hypergraph)
        m_array.append(m)
        rho_array.append(rho)
        
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
    
    for i in range(num_simulations):

        hypergraph_init=copy.deepcopy(hypergraph) #after each simulation, we want the initial hypergraph to be the given one
        opinions_init=copy.deepcopy(opinions_hypergraph)
        op_dict_init=copy.deepcopy(op_dict)

        num_sp_array[i], size_var, ratio_ones_array[i], m_array, rho_array =statistics_calc(hypergraph_init, opinions_init, op_dict_init,num_it,gamma)

        m_multi_array.append(m_array)
        rho_multi_array.append(rho_array)
        
        size_array.extend(size_var)
        # print(rho_multi_array)
        # print("-----------")        
    return num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
    
def figure_generator(num_sp_array,size_array,ratio_ones_array): 
    """"Plots the histograms of the multiple simulations"""
    #bins = range(10)
    
    fig, ax = plt.subplots(1, 3)


    ax[0].hist(num_sp_array, bins=np.arange(min(num_sp_array), max(num_sp_array) + 0.5, 0.5))
    # bins_labels(bins, fontsize=7)
    
    ax[1].hist(size_array,bins=np.arange(min(size_array), max(size_array) + 0.5, 0.5))
    # bins_labels(bins,fontsize=7)
    
    ax[2].hist(ratio_ones_array,bins=np.arange(min(ratio_ones_array), max(ratio_ones_array) + 0.005, 0.005))
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
    num_sp_simulations=[0]*num_intervals
    cmap = plt.cm.get_cmap('hsv', num_intervals+1)
    
    for i in range(num_intervals):
        num_sp_array,size_array,ratio_ones_array, magnetization_simulations[i], rho_simulations[i]=statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma[i])


        #num_sp_simulations[i]=sum(num_sp_array)/len(num_sp_array)
        for j in range(num_simulations):
            
                #plt.scatter(rho_simulations[i][j],magnetization_simulations[i][j],s=2,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "")
                plt.scatter(rho_simulations[i][j],num_sp_simulations[i][j],s=2,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "")

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("Density of inactive edges")
    plt.ylabel("Magnetization")
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

def parabola(x, a, b,c):
	return a * (x-c)**2 + b

def fit_m_vs_density(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, gamma):
    """Calculates coefficients of a fitted parabola and plots it on the m vs rho diagram"""
    num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array=statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma)
    y_coords=list(flatten(m_multi_array))
    x_coords=list(flatten(rho_multi_array))
    points=list(zip(x_coords, y_coords))
    sorted(points , key=lambda k: [k[1], k[0]])
    points=np.array(points)
    y_coords_fit=points[:,1]
    x_coords_fit=points[:,0]
    params = curve_fit(parabola, y_coords_fit, x_coords_fit)
    a,b,c=params[0]
    
    y=np.linspace(min(y_coords),max(y_coords),1000)
    x=parabola(y,a,b,c)
    
    print("The coefficients of the parabola are: a={}, b={}, c={}" .format(a,b,c))
    plt.scatter(x_coords,y_coords,s=2,color="orange")
    plt.plot(x, y,color="black")
    plt.show()
    return

    

    


    
#------------------------Parameters------------------------------#

N=300 #number of nodes
S=40 #size of hyperedges
n=40 #number of hyperedges
gamma=0.49 #threshold for split
num_it=1000 #number of iterations 
num_simulations=200 #number of simulations for statistics
num_intervals=2 #number of different gamma values for magnetization vs density function
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


# num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array=statistics_calc_multi(num_simulations,H, opinions_H, op_dict,num_it,gamma)
magnetization_vs_density(num_simulations,H, opinions_H, op_dict,num_it, num_intervals)

# fit_m_vs_density(num_simulations,H, opinions_H, op_dict,num_it,gamma)

# print(num_sp_array,size_array)
# print("The number of species for the simulations {:}" .format(num_sp_array))
# print("The total number of populations of the species for the simulations {:}" .format(size_array))
# print("The ratio of the opinions of the species {:}" .format(ratio_ones_array))

# print(num_sp_array)
# print(size_array)
# print(ratio_ones_array)

# figure_generator(num_sp_array,size_array,ratio_ones_array)
