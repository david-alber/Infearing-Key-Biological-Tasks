# -*- coding: utf-8 -*-
"""DecayLog3A.ipynb


### Work flow: 
1.  Preprocess Data
2.  Do PCA analysis
3.  Get rid of outliers
4.  Find Archetypes in Gene_space
5.  Convert the Archetype coordinates from Gene- to Go- space
5.  Depict the spectrum of each archetype in Go-space. (Which features are enhanced?)
6.  !Do detailed decay analysis with slope and p- value!

Here we want to consider 3 Archetypes
"""

import sys
sys.path.append("C:/Users/alber/OneDrive/Dokumente/SCRIPTS/PYTHON/ML/SciLifeProject")
from parti_lib import *
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #Progressbar
import time
import seaborn as sns

#%% Load data
gene_data = pd.read_csv("C:/Users/alber/OneDrive/Dokumente/SCRIPTS/PYTHON/ML/SciLifeProject/Data/data_RNA_Seq_expression_cpm.txt",delimiter= "\t")
gene_data = gene_data.transpose()

names = np.arange(0,160)#place holders to seperate gene_expr in GoData-Matrix
GoData = pd.read_csv("C:/Users/alber/OneDrive/Dokumente/SCRIPTS/PYTHON/ML/SciLifeProject/particode/MSigDB/c2.cp.v4.0.symbols.gmt",
                     delimiter="\t", names = names)
GoData = GoData.drop(columns = 1,axis = 1)

#%%  Preprocess

[data_scaled,gene_names] = preprocess_data(gene_data,logarithmic=True,scaling = "mean")

#%% PCA Analysis"""

pca_analysis = pca_ana1(data_scaled,confidence = "none",
                        n_components = 50,plot = True)

#%% Visualize the Gene PCA reduced Data -> Reduce to 2 components
#PCA Transfrom 2 data sets. One for visualization and one for analysis.
#For 3 Archetypes forming a triangle we consider the projection on 2D

#For visualization
[red_data_scaled_plot,pca_plot] = pca_trafo(data_scaled,n_components=2) 
#For analysis
[red_data_scaled,pca]           = pca_trafo(data_scaled,n_components=40)
pca_data_plot(red_data_scaled_plot)

#%% Depict data in 2D and detect outliers
#Detect outliers in first 2 PCA- components and delete them from the sample data set if needed.
# =============================================================================
# [o1,o2]              = outliers(red_data_scaled_plot,percentile = 0.8,plot = True)
# print(red_data_scaled_plot.shape)
# o1 = [] #If there are no outliers
# data_scaled1            = kill_outlier(data_scaled,o1)
# red_data_scaled1        = kill_outlier(red_data_scaled,o1)
# red_data_scaled1_plot   = kill_outlier(red_data_scaled_plot,o1)
# red_data_scaled1_plot = red_data_scaled_plot
# print(red_data_scaled_plot.shape)
# =============================================================================

#%% KMeans Analysis 
#How many clusters should be considered, based on average s-value.
#Perform this analysis on the larger PCA-data set, not the plotting one.
data_scaled1 = data_scaled
red_data_scaled1 = red_data_scaled
red_data_scaled1_plot = red_data_scaled_plot
best_n_clusters = k_means_ana(red_data_scaled1,max_n_clusters=20,plot=True)

#%% K-means clustering
#Based on the plot above, select the ideal number of clusters and depict the data set in 2 PCA- components.

n_clusters = 18 #based on k_means_ana()
centers_plot = k_means_trafo(red_data_scaled1_plot,n_clusters=n_clusters,plot=True)
#For the analysis space
centers = k_means_trafo(red_data_scaled1,n_clusters=n_clusters,plot = False)

#%% Archetype Analysis
max_noc = 9 #Maximum number of archetypes to investigate
var_ana = archetype_ana(max_noc,red_data_scaled1,delt = 0.4,plot = True)

"""
#### Select number of components and compute positions of Archetypes
This can be based on k-means analysis, so how many clusters we found or based on Archetype analysis, i.e. plot above.

Note: Delta can be $0\leq \delta \leq 1$ and the closer it is to 1 the more *slack* do we consider, i.e. our polytope gets bigger.
"""
#%% PCA Analysis
noc = 3#based on archetype_ana()
delta = 0.8
#noc = n_clusters #based on and k-means
#Plot archetypes and sample points in 2D PCA-components
XC_plot, S_plot, C_plot, SSE_plot, varexpl_plot = PCHA(red_data_scaled1_plot.T, noc=noc, delta=delta)
plotArch(red_data_scaled1_plot,XC_plot,noc)
#Archetypes with m_characteristics > 3
XC, S, C, SSE, varexpl = PCHA(red_data_scaled1.T, noc=noc, delta=delta)



"""
#### Transform Archetypes in Gene_space to Go_space coordinates
That is, map A[n_samples,m_genes] -> A[n_samples,GO_tasks]
"""
#%% Go Trafo
A = np.squeeze(np.asarray(XC))#cast to array form
#Position of Archetypes in full gene space
A_gene_space = pca.inverse_transform(A.T)
#centers_gene_space = pca.inverse_transform(centers)
# If 20 gene family members are NOT found in the gene_data, then Go-task is killed!
[A_go,go_names]       = getGo(A_gene_space,gene_names,GoData,matchlimit = 20)
[go_data,go_names]    = getGo(data_scaled1,gene_names,GoData,matchlimit = 20)


"""
#### Importance of archetypes:
Calculate the normalized distances of all samples with respect to each archetype and depict the result in form of a histogram. (The same thing can be done for the cluster centers.)
"""
#%% Distances to archetype
#percentile = 0.1 means 10% of data -> approx 45 sample points
dists = dist2A(A_go,go_data,percentile = 0.1,plot = True,histtype = "step")
#dists_k_centers = dist2A(centers_go_space,go_data,percentile = 0)

"""
####  Which Go- expressions are strongly represented in the Archetypes
We show the spectrum of each archetype and how strong the Go- exressions are represented respectively.
"""
#%% I-Max
i_max,i_max_names,i_sorted = gene_expr_A(A_go,go_names,percentile=0.85,plot = True,rows = 2,cols = 2)
print(i_max_names)

"""
#### Decay of features with respect to distance
Observe the decrease of the selected feature with respect to distance to one archetype.
"""

atype = 0#Which archetype,i.e. row of i_max
featuretype = 0#Spectrum index of first maximum, from i_max
decay_max = feature_decay(go_data,go_names,dists,i_max,atype,featuretype,plot = True)
print(f"Decay of gene feature {go_names.iloc[i_max[atype,featuretype]]} with respect to normalized distance"
      f"from archetype {atype} is under consideration.\n")

"""
#### Block average the feature decay with respect to distance.
That is, we create blocks of data, and within each block the amount of Go-expression get´s averaged. This smoothens out the decay of this Go-expression with respect to the normalized distance of the archetype
"""

#Consider the maximally expressed feature of the archetype
blocks = 10 #number of blocks to compress data, i.e. 10% of data for each block
blockvec,blocktt,error,sigmaB,bdata = blocked_feature_decay(decay_max,blocks,dists,go_names,
                                                            i_max,atype,featuretype,plot=True)
print(f"Each data point in blocked_feature_decay represents {blocks:.1f}% of total data.\n"\
      f"Here we have {np.shape(go_data)[0]/blocks:.1f} samples per block.\n")


"""
#### Linear regression and P-value analysis for all Go-expressions
Using the Block averaged feature-decay, we want to find the Go-task that thas the steepest negative slope as the distance from the Archetype increases. We therefore sample over all Go-expressions and compute the slope and the respective p-value. Note that this should roughly match with the maximally expressed Go-expressions in the Archetype spektra.
"""
#%% C-P Ana
stbin_size = 1#Corresponding to 10% of data.
C_Matrix,c_min_index,P_Matrix,p_min_index,all_features = c_p_matrix(stbin_size,
                                                            go_data,dists,noc,bdata,go_names)
c_indices_sorted = np.argsort(C_Matrix,axis =1)
p_indices_sorted = np.argsort(P_Matrix,axis =1)


"""
### Heat Map:
We look at the heat map of the coefficients that we were fitting with linear regression. 
This shows the full spectrum of the Go-expressions and how important they are, i.e. darker areas suggest that this Go-expression is important for this archetype and brighter areas suggest that they are not so important for this archetype. 

With this we can see *exclusive* features. That is when an Archetype shows one *strong expression* which is a *weak expression* for the other Archetypes
"""
#%% Heatmap
fig, ax = plt.subplots(figsize=(12,10)) 
sns.heatmap(C_Matrix,cbar = True,ax=ax)#cbar_kws=None
#Show the first 3 most negative slopes for each archetype
show_minima(C_Matrix)

for i in range(0,np.shape(i_max)[0]):
    print(f"c_min_index A{i}:\t {c_indices_sorted[i,:8]} \n")
    print(f"i_max of A{i}:\t {i_sorted[i,:8]} \n")
    print(f"p_min_index A{i}:\t {p_indices_sorted[i,:8]} \n")
    
#%% CIP - List & common features    
cip_list = cip_listing(noc,go_names,C_Matrix,c_indices_sorted,A_go,i_sorted,
                       P_Matrix,p_indices_sorted,save=True,filename='CIP4-list.csv')        

block_feature_decay(go_data,go_names,dists,c_min_index)

#common_expressions(c_indices_sorted,i_sorted,go_names,first_n = 20)

