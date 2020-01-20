# -*- coding: utf-8 -*-
"""ParTI_lib.ipynb


# This is the function liberary to the ParTI project

### Instructions

Load in working functions and attach a doc- string to them describing what they do. Or use a text field to describe. 

If needed download an up to date version of this file as .py file to your local machine and implement it to the current working file. 

To use on your local machine: `File` --> `Download .py` and save as `other` to some folder.  Then 
`import sys`

`sys.path.append("filemapth")`

`from parti_lib import *`


[Pareto Based Multiobjective Machine Learinging](https://drive.google.com/drive/u/0/folders/1kP5DvV4LfFO7R63IYTo5LohTvHJ2LJSx)

[Inferring biological tasks
using Pareto analysis of
high-dimensional data](https://drive.google.com/drive/u/0/folders/1kP5DvV4LfFO7R63IYTo5LohTvHJ2LJSx)

[Evolutionary Trade-Offs, Pareto
Optimality, and the Geometry
of Phenotype Space](https://drive.google.com/drive/u/0/folders/1kP5DvV4LfFO7R63IYTo5LohTvHJ2LJSx)

[Pareto Task Inference (ParTI) method](https://www.weizmann.ac.il/mcb/UriAlon/download/ParTI)
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from py_pcha import PCHA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy as sp
from sklearn.linear_model import LinearRegression
from tqdm import tqdm #Progressbar
import time

def pca_data_plot(red_gene_space_plot):
    """
    Void function to plot the PCA reduced data.
    IN: array[n_samples,m_cahracteristics = 3]
    
    """
    if np.shape(red_gene_space_plot)[1] == 3: 
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(red_gene_space_plot[:,0], red_gene_space_plot[:,1],
                   red_gene_space_plot[:,2],c = "k", s=5,label = "All points")
        ax.set_title("Data in 3 PCA components")
        ax.set_xlabel('1st pca component')
        ax.set_ylabel('2nd pca component')
        ax.set_zlabel('3rd pca component')
        plt.legend(loc = "best")
        plt.tight_layout()
        plt.show()
    if np.shape(red_gene_space_plot)[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(red_gene_space_plot[:,0], red_gene_space_plot[:,1],
                   c = "k", s=5,label = "All points")
        ax.set_title("Data in 2 PCA components")
        ax.set_xlabel('1st pca component')
        ax.set_ylabel('2nd pca component')
        plt.legend(loc = "best")
        plt.tight_layout()
        plt.show()

def preprocess_data(data,logarithmic,scaling):
    """
    Convert the data to a preprocessed numpy array.
    IN:data[n_samples,m_characteristics] = can be np.array. If not it will be converted to an np.array.
       logarithmic: Apply an element wise logarithmic scaling. All data points are shifted by +1 s.t: log(1) = 0
       scaling = 'std','normalize','none'. The 'std' scaling subtracts the sample mean from every sample point and
                 rescales s.t: the variance = std deviation of the sample point values = 1.
                 The 'normalize' scaling scales every value with the l2 norm s.t: xi -> xi / (sum(xi^2)^0.5).
    """
    gene_names = data.iloc[0]
    gene_names = gene_names.to_numpy()
    gene_names = gene_names.astype(str)
    data_m = data.iloc[2:]
    data_m = data_m.to_numpy()
    data_m = data_m.astype(float)
    #Apply Logrithmic Scaling to data
    if logarithmic == True:
        data_m = np.log(data_m + 1) #element wise// lift data s.t: 0->1 and log(1) = 0
        
    if  scaling == 'std':
        data_scaled = preprocessing.scale(data_m,axis=0)
        
    if  scaling == 'normalize':
        data_scaled = preprocessing.normalize(data_m,norm='l2')

    if scaling == 'mean': 
      #subtract the mean of each row from each element of the row
      data_scaled = data_m - data_m.mean(axis = 0)
        
        
    if  scaling == 'none':
        data_scaled = data_m
             
    return(data_scaled,gene_names)

def pca_ana1(data_matrix,confidence,n_components,plot):
    """
    call: pca_analysis = pca_ana1(data_matrix,confidence='none',n_components=12,plot=True)
    IN: confidence   = 0-1 or 'none': Fix a confidence -> n_components is adjusted
        n_components = integer: Fix number of pca components -> confidence is adjusted
        data matrix: rows -> samples; columns -> characteristics
    OUT: data_matrix_PCA[n_samples,n_components]:n_samples: number of samples; 
         n_components: reduced number of characteristics. The data is in the PCA SPACE
         if plot = True, then a plot is outputed
    """
    #Do pca ana for fixed number of components
    if (confidence == 'none'):
        components_range = np.arange(2,n_components+1)
        variance_range = np.zeros(len(components_range))
        for i in range(len(components_range)):
            pca = PCA(n_components = components_range[i])
            data_matrix_PCA = pca.fit_transform(data_matrix)
            variance_sum = sum(pca.explained_variance_ratio_)
            variance_range[i] = variance_sum 
            
    if n_components == 'none':
        pca = PCA(confidence)
        data_matrix_PCA = pca.fit_transform(data_matrix)
        print('PCA-ANA: %.3f of the information is explained in %.1f components' %(confidence,len(data_matrix_PCA[1])))

    if (plot == True and confidence == 'none'): 
        plt.figure()
        plt.plot(components_range,variance_range,"*-")
        plt.title("PCA- analysis",fontsize = 18);
        plt.ylabel("Percent of information",fontsize = 16);
        plt.xlabel("Number of components",fontsize = 16)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return(data_matrix_PCA)

def pca_ana(data_matrix,plot = False):
    """
    call: pca_analysis = pca_ana(gene_data,plot = True) 
    Input = data matrix: rows -> samples; columns -> characteristics
    Output= 2 x 9 array.dim 0 -> [percentile of information, number of components]; dim 1 -> respective data points
    if plot = True, then a plot is outputed
    In order to fix a certain number of components: 
    pca = sklearn.decomposition.PCA(n_components)  ; pca.fit(data_matrix); red_space = pca.transform(data_matrix)
    """
    n_components = np.zeros(10)
    percentiles = np.linspace(0.1,1,10)
    for i in range(1,len(percentiles)):
        pca = sk.decomposition.PCA(0.1*i)
        pca.fit(data_matrix)
        n_components[i-1] = len(pca.explained_variance_ratio_)
    pca_ana = np.concatenate(([percentiles[:-1]],[n_components[:-1]]),axis = 0)
    if plot == True: 
        plt.figure()
        plt.plot(pca_ana[1],pca_ana[0],"*-")
        plt.title("PCA- analysis",fontsize = 18);
        plt.ylabel("Percentile of information",fontsize = 16);
        plt.xlabel("Number of components",fontsize = 16)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return(pca_ana)

def pca_trafo(data_matrix,n_components):
  """
  After analyzing the principal components transform the full data matrix in 
  reduced space according to the components one wants. 
  In:   1.data_matrix (m x n ), m = samples and n = charachteristics
        2. Number of components under consideration (integer). 
        Choose it according to the analysis from pca_ana(.) function.
  Out: array    = reduced matrix
       function = pca
  """
  pca = sk.decomposition.PCA(n_components)
  pca.fit(data_matrix)
  red_space = pca.transform(data_matrix)
  print(f"Total information in {n_components} principle components {sum(pca.explained_variance_ratio_)}")
  return(red_space,pca)

def outliers(red_gene_space,percentile,plot = True):
    """
    Depict data in 3D and detect outliers.
    In:     1. data[s_samples,3_main PCA features]
            2. Which 0<percentile<=1 of maximum for each component is considered an outlier
            3. Want to plot? 
    Out:    Three boolean matrices where the outlier with respect to each pca component is true 
    """
    outliers_1max = red_gene_space[:,0]>= percentile*max(red_gene_space[:,0])
    outliers_1min = red_gene_space[:,0]<= percentile*min(red_gene_space[:,0])
    true_1max = np.where(outliers_1max == True)[0]
    true_1min = np.where(outliers_1min == True)[0]
    outliers_1 = np.append(true_1max,true_1min)
    
    outliers_2max = red_gene_space[:,1]>= percentile*max(red_gene_space[:,1])
    outliers_2min = red_gene_space[:,1]<= percentile*min(red_gene_space[:,1])
    true_2max = np.where(outliers_2max == True)[0]
    true_2min = np.where(outliers_2min == True)[0]
    outliers_2 = np.append(true_2max,true_2min)
    
            
    print(outliers_2.shape)
    if np.shape(red_gene_space)[1] == 3:
        outliers_3max = red_gene_space[:,2]>= percentile*max(red_gene_space[:,2])
        outliers_3min = red_gene_space[:,2]<= percentile*min(red_gene_space[:,2])
        true_3max = np.where(outliers_3max == True)[0]
        true_3min = np.where(outliers_3min == True)[0]
        outliers_3 = np.append(true_3max,true_3min)

    if plot == True:
        fig = plt.figure()
        if np.shape(red_gene_space)[1] == 3: 
            print("Outliers with respect to first 3 PCA- componets. \n"
              "Outliers in 1st PCA component is most important!")
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(red_gene_space[:,0], red_gene_space[:,1], red_gene_space[:,2], c = "k", s=5,label = "All points")
            ax.scatter(red_gene_space[outliers_2,0], red_gene_space[outliers_2,1], red_gene_space[outliers_2,2], c = "g",label = "Outliers in 2nd PCA" )
            ax.scatter(red_gene_space[outliers_3,0], red_gene_space[outliers_3,1], red_gene_space[outliers_3,2], c = "c", label = "Outliers in 3rd PCA")
            ax.scatter(red_gene_space[outliers_1,0], red_gene_space[outliers_1,1], red_gene_space[outliers_1,2], c = "b",label = "Outliers in 1st PCA")
            ax.set_title("Data in 3 PCA components and outliers",fontsize = 16)
            ax.set_xlabel('1st pca component',fontsize = 16)
            ax.set_ylabel('2nd pca component',fontsize = 16)
            ax.set_zlabel('3rd pca component',fontsize = 16)
            plt.legend(loc = "lower left")
            plt.tight_layout()
            plt.show()
            return(outliers_1,outliers_2,outliers_3)
        if np.shape(red_gene_space)[1] == 2:
            print("Outliers with respect to first 2 PCA- componets. \n"
              "Outliers in 1st PCA component is most important!")
            ax = fig.add_subplot(111)
            ax.scatter(red_gene_space[:,0], red_gene_space[:,1],c = "k", s=5,label = "All points")
            ax.scatter(red_gene_space[outliers_2,0], red_gene_space[outliers_2,1], c = "g",label = "Outliers in 2nd PCA" )
            ax.scatter(red_gene_space[outliers_1,0], red_gene_space[outliers_1,1], c = "b",label = "Outliers in 1st PCA")
            ax.set_title("Data in 2 PCA components and outliers",fontsize = 16)
            ax.set_xlabel('1st pca component',fontsize = 16)
            ax.set_ylabel('2nd pca component',fontsize = 16)
            plt.legend(loc = "lower left")
            plt.tight_layout()
            plt.show()
            return(outliers_1,outliers_2)

def kill_outlier(red_gene_space,outliers): 
    """
    After outlier detection with outliers(), use this func to delete  
    thoses samples from data set.
    In:     1. red_gene_space[s_samples,n_features] in full space
            2. outliers are with respect to one specific PCA component. 
                (If you want to kill all --> call the function 3 times with o1,o2,o3)  
    Out: red_gene_space[s-outliers,n_features]
    """
    o_index = np.where(outliers== True)[0]
    print(f"Sample index of outlier(s) that are going to be removed {o_index}")
    outlier_free_genes = np.delete(red_gene_space,o_index,axis = 0)
    return(outlier_free_genes)

"""[PCHA](https://github.com/ulfaslak/py_pcha) Python package. Read description in the link and `pip install py_pcha`"""

# XC, S, C, SSE, varexpl = PCHA(red_gene_space1.T, noc=4, delta=0.1)

def verts_list(XC):
    """
    Make lists of the vertices that form sides of the polytope
    Depending on how many vertices we have.
    Note: the coordinates of XC should be in 3 principal components
    in:    XC = output of PCHA Position of Archetypes
    """
    #3 Vertices -> Trinagle 1 side
    if np.shape(XC)[1] == 3:
        verts = [[XC[:,0],XC[:,1],XC[:,2]]]
    #4 Vertices -> Pyramide 4 sides
    if np.shape(XC)[1] == 4:
        verts = [[XC[:,0],XC[:,1],XC[:,2]],[XC[:,1],XC[:,2],XC[:,3]]
                 ,[XC[:,0],XC[:,2],XC[:,3]],[XC[:,0],XC[:,1],XC[:,3]]]
    #5 Vertices -> 5 sides
    if np.shape(XC)[1] == 5: 
        A = np.squeeze(np.asarray(XC))
        print(np.shape(A))
        z_max = max(A[2,:])
        print(z_max)
        a_top = np.where((A[2,:]== z_max))[0][0]
        others = np.delete(np.arange(0,5),a_top)
        print(others)
        print(a_top)
        verts = [[XC[:,a_top],XC[:,others[0]],XC[:,others[1]]],
                 [XC[:,a_top],XC[:,others[2]],XC[:,others[3]]],
                 [XC[:,a_top],XC[:,others[0]],XC[:,others[2]]],
                 [XC[:,a_top],XC[:,others[1]],XC[:,others[3]]],
                 [XC[:,others[1]],XC[:,others[0]],XC[:,others[2]],XC[:,others[3]]]]
    #6 Vertices -> 8 sides
    if np.shape(XC)[1] == 6: 
        A = np.squeeze(np.asarray(XC))
        verts = [[A[:,1],A[:,4],A[:,5]],
         [A[:,1],A[:,5],A[:,3]],
         [A[:,1],A[:,3],A[:,2]],
         [A[:,1],A[:,0],A[:,2]],
         [A[:,1],A[:,0],A[:,4]],
         [A[:,2],A[:,0],A[:,4],A[:,5],A[:,3]]]
        
    return(verts)

def plotArch(red_data,XC,noc):
    """
    Visualize the position of archetypes and sample points in 3PC- projection
    Note: For noc > 5 the verts_list(XC) function might has to be updated
    in:     1. red_data[i,j]; i = samples, j = principle components (coordinates)
            2. XC = output of PCHA Position of Archetypes
            3. Number of components >= 6
    """
    n_archetypes = np.shape(XC)[1]
    dim = np.shape(XC)[0]
    archColors = cm.plasma(np.arange(n_archetypes).astype(float) / (n_archetypes))
    

    if dim == 3:
        verts = verts_list(XC)
        srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
        fig = plt.figure()
        #if np.shape(red_data)[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(red_data[:,0], red_data[:,1], red_data[:,2], c = "k", s=5)
        #ax.scatter(XC[0,:],XC[1,:],XC[2,:],c = "k",s=70)
        for i in range(0,noc): 
            ax.scatter(XC[0,i],XC[1,i],XC[2,i],s=70,c=archColors[i,:],label = "A"+str(i))  
        ax.add_collection3d(srf)
        ax.set_xlabel('1st pca component',fontsize = 14)
        ax.set_ylabel('2nd pca component',fontsize = 14)
        ax.set_zlabel('3rd pca component',fontsize = 14)
        ax.legend()
        plt.tight_layout()
        plt.show()
    if dim == 2 and n_archetypes == 3:
        verts = np.squeeze(np.asarray(verts_list(XC)))
        verts = np.reshape(verts,(-1,2))
        verts = list(verts)

        surf = plt.Polygon(XC.T, color='#800000',alpha=.25)
        fig = plt.figure()
        #if np.shape(red_data)[1] == 3:
        ax = fig.add_subplot(111)
        ax.scatter(red_data[:,0], red_data[:,1], c = "k", s=5)
        for i in range(0,noc): 
            ax.scatter(XC[0,i],XC[1,i],s=70,c=archColors[i,:],label = "A"+str(i))  
        plt.gca().add_patch(surf)
        ax.set_xlabel('1st pca component',fontsize = 14)
        ax.set_ylabel('2nd pca component',fontsize = 14)
        ax.legend()
        plt.tight_layout()
        plt.show()

def archetype_ana(max_noc,red_data,delt = 0.1, plot = False):
    """
    Analyze which number of vertices is ideal to represent the distinct 
    features of the data
    in:     1. Maximum number of archetypes to be considered
            2. reduced data[i,j]; i = samples; j = dimensions in pca space
            3. Which delta 0<delta<1 to consider. How much "slack" for polytope
            4.Do you want to plot: True
    out:    data[i,j]; i = number of vertices; j = variation explained
    """
    var_arr = np.zeros(max_noc)
    nocv = np.arange(1,max_noc+1)
    for i in range(1,max_noc+1): 
        XC, S, C, SSE, varexpl = PCHA(red_data.T, noc=nocv[i-1], delta=delt)
        var_arr[i-1] = varexpl
    data = np.concatenate(([nocv],[var_arr]),axis = 0)
    print(data.shape)
    if plot == True: 
        plt.figure()
        plt.plot(data[0,:],data[1,:],"*-")
        plt.title(f"Archetype analysis for data with {len(red_data[1]):d} characteristics",fontsize = 16);
        plt.ylabel("Variation explained",fontsize = 16);
        plt.xlabel("Number of Vertices",fontsize = 16)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return(data)

def dist2A(Apos,data,percentile,plot=False,histtype = "step"):
    
    """
    Compute the of all points to each archetype
    in: 1.Apos[j,k] = position of all j archetypes and k dimensions
        2.data[i,k] = all i- sample points in k- dimensions
        3.Do you want to plot: True
        4. Which type of histogram: "bar", "stepfilled"
    out: dists[j,i] = distance of the i´th data point to the j´th archetype
    """ 
    #loop over all samples
    samples = np.shape(data)[0]
    n_a = np.shape(Apos)[0]
    dists = np.zeros((n_a,samples))
    archColors = cm.plasma(np.arange(n_a).astype(float) / (n_a))
    for j in range(0,n_a):
        for i in range(0,samples):
            dists[j,i] = np.linalg.norm(np.abs(data[i,:]-Apos[j,:]))
        dists[j,:] = dists[j,:]/max(dists[j,:])
    if plot == True: 
        plt.figure()
        plt.title("Amount of points vs distance to archetype",fontsize = 18)
        plt.xlabel("Normalized distance",fontsize = 16); plt.ylabel("Amount of sample points",fontsize = 16)
        for k in range(0,n_a):
            plt.hist(dists[k,:],int(percentile*np.shape(dists)[1]),
                     color =archColors[k,:],histtype=histtype,label = "A"+str(k))
        plt.legend()
        plt.tight_layout()
        plt.show()
    return(dists)

def gene_expr_A(A_gene_space,gene_names,percentile,plot= False,rows =2, cols =2):
    """
    Finding the maxima in gene expressions, i.e. where in the gene space
    do we find maxima, if we look at the Archetypes
    in:     1. A_gene_space[i,k]; i = number of archetype; k = gene space
            2. Names of all the gene expressions
            3. Percentile <=1; which percentile of the maximum is considered
            4. Do you want to plot the histogram?
            5.Number of rows in the figure
            6.Number of columns in the figure
    out:    A_i_maxd[i,k]: For archetype i, which genes k are over represented
            This matrix has zeros, to match dimensions!
    """
    g_dim = np.shape(A_gene_space)[1]
    n_A = np.shape(A_gene_space)[0]
    i_max = np.zeros((n_A,30))
    A_i_max = np.zeros((n_A,30))
    A_i_names = [] #Output the names of the corresponding attributes with gne_names
    #Minus because we want the biggest first and argsort(.) is in ascending order
    sorted_expressions = np.argsort(-A_gene_space,axis = 1)
    archColors = cm.plasma(np.arange(n_A).astype(float) / (n_A))
    #Evaluate maxima
    for i in range(0,n_A):
        i_max = np.where(A_gene_space[i,:]>=percentile*max(A_gene_space[i,:]))[0]
        #Put maxima in a list
        for k in range(0,len(i_max)):
            A_i_max[i,k] = i_max[k]
        non0 = A_i_max[i,np.nonzero(A_i_max[i,:])][0]
        non0 = non0.astype(int)
        for j in range(0,len(non0)):
            gene_index = non0[j]
            A_i_names = np.append(A_i_names,(gene_names.iloc[gene_index],"\t"))
        A_i_names = np.append(A_i_names,"\n")    
    if plot == True:
        g_space = np.arange(0,g_dim)
        fig = plt.figure()
        for i in range(0,n_A):
            non0 = A_i_max[i,np.nonzero(A_i_max[i,:])][0]
            non0 = non0.astype(int)
            ax = fig.add_subplot(rows,cols,i+1)
            ax.scatter(g_space,A_gene_space[i,:],c="k")
            ax.scatter(non0,A_gene_space[i,non0],c=archColors[i,:])
            ax.set_xlabel("Expression Spectrum")
            ax.set_ylabel("Amount")
            ax.set_title(f"Archetype {i}")
        fig.tight_layout()
    return(A_i_max.astype(int),A_i_names,sorted_expressions)

def feature_decay(gene_data_m,gene_names,dists,i_max,atype,featuretype,plot = False):
    """
    Decay of a specific feature in the samples, that is maximally represented in the position
    of the Archetype, with respect to the normalized distance to the Archetype. 
    In:     1. gene_data_m[i,j] matrix of gene data. i = samples; j = characteristics
            2. gene_names[j] list of the gene names
            3. dists is output of dist2A(.)- function, i.e. for each sample point the distance to respective archetype
            4. i_max is output of gene_expr_A(.), i.e array of indices of maximally expressed gene characteristics of archetypes
            5. atype = n = integer. Which archetype to look at (0,1,2,...)
            6. featuretype = which of the maximized features to look at. Which of the indices m from i_max[n,m]
            7. Do you want to plot? True
    Out: decay[2,i]: A list of sample points where decay[1,i] are the distances to archetype n and decay[2,i] is the decay of the feature under investigation  
    """
    n_a = np.shape(dists)[0]
    archColors = cm.plasma(np.arange(n_a).astype(float) / (n_a))
    a_sorted = np.zeros_like(dists)
    for i in range(0,n_a):
        a_sorted[i,:] =  np.argsort(dists[i,:])
    a_sorted = a_sorted.astype(int)
    featuredecay = gene_data_m[a_sorted[atype,:],i_max[atype,featuretype]]
    featuredist  = dists[atype,a_sorted[atype,:]]
    name = gene_names.iloc[i_max[atype,featuretype]]
    decay = np.concatenate((featuredist,featuredecay),axis = 0)
    decay = np.reshape(decay,(2,len(featuredecay)))
    #print(f"Decay of gene feature {name} with respect to normalized distance from archetype {atype} is under consideration.")
    if plot == True: 
        archColors = cm.plasma(np.arange(n_a).astype(float) / (n_a))
        plt.figure()
        plt.plot(featuredist,featuredecay,"*-",c = archColors[atype,:])
        plt.suptitle(f"Decay with respect to distance of A{atype}",fontsize = 14 )
        plt.title(f"{name}",fontsize=8)
        plt.xlabel(f"Normalized distance of samples to A{atype}",fontsize = 14)
        plt.ylabel(f"Amount task/gene in samples",fontsize = 14)
        plt.show()
    return(decay)

def getGo(gene_data,gene_names,GoData,matchlimit = 20):
    """
    names = np.arange(0,160)
    GoData = pd.read_csv("C:/Users/lUKAS/Documents/ParetoOptimization/particode/MSigDB/c2.cp.v4.0.symbols.gmt",
                     delimiter="\t", names = names)
    GoData = GoData.drop(columns = 1,axis = 1)
    Sample call: 
    [Go,task_names] = getGo(data_scaled,gene_names,GoData,matchlimit = 20)
    
    First:  [data_scaled,gene_names] = preprocess_data(gene_data,logarithmic=False,scaling = "none"), then
    Call :  [Go,task_names] = getGo(data_scaled,gene_names,GoData)
    in:     1. gene_data[s_samples,j_characteristics] as numpy float array. 
            2. gene_names numpy str. All the names, as strings from the gene_data
            3. GoData[t_tasks,j_genes] pandas data frame with t- tasks and j_gene names
            4. integer: After how many "not found gene expressions" do we disregard a GO-task
    out: Go[s_samples,t_tasks], i.e. how strong each Task is expressed (average) in each sample
    """
    Go = np.zeros((gene_data.shape[0],GoData.shape[0]))
    task_names = GoData.iloc[:,0]
    count = 0
    start_time = time.time()
    for t in tqdm(range(0,GoData.shape[0]),position = 0,desc = "GO-tasks"):#all tasks GoData.shape[0]
        GoData1 = GoData.iloc[t,1:]
        GoData1 = GoData1.dropna()
        GoData1 =  GoData1.to_numpy()
        GoData1 = GoData1.astype(str)
        match_list = []
        for i in range(0,GoData1.shape[0]):#Which genes match
            match = np.where(GoData1[i]==gene_names)[0]
            if match.size == 0: #count the ones that do not match
                count +=1
                temp = t
            if match.size > 0:
                match_list = np.append(match_list,int(match))
        match_list = match_list.astype(int)
        for s in range(0,gene_data.shape[0]):#each task for all samples
            Go[s,t] = np.mean(gene_data[s,match_list])
            if count > matchlimit:#More than 20 genes not found for GO-task `temp`
                Go[s,temp] = 0
    end_time = time.time()
    print(f"Elapsed time = {end_time-start_time:.2f}")
    [Go,task_names] = reduced_Go(Go,task_names)
    print(f'gene_data[{len(gene_data[:,0]):d}_samples,'+ f'{len(gene_data[0,:]):d}_genes]',  
          f'was converted to go_data[{len(Go[:,0]):d}_samples,{len(Go[0,:]):d}_tasks] ')
    return(Go,task_names)

def reduced_Go(Go,task_names):
    samples = np.shape(Go)[0]
    index = []
    for t in range(0,np.shape(Go)[1]):
        if Go[0,t] != 0: #if sample 0 shows 0 then all the other samples have 0 to
            index = np.append(index,t)
    relevant_t_names = task_names.iloc[index]
    Go1 = Go[Go != 0]
    Go1 = np.reshape(Go1,(samples,-1))
    return(Go1,relevant_t_names)

"""### K-means analysis functions

[Selecting the number of clusters with silhouette analysis on KMeans clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)
"""

def k_means_ana (data,max_n_clusters,plot):
    """
    Analyses the best number of clusters according to the average silhouette score. 
    Silhouette coefficients (as these values are referred to as) near +1 indicate 
    that the sample is far away from the neighboring clusters. A value of 0 indicates 
    that the sample is on or very close to the decision boundary between two neighboring clusters
    and negative values indicate that those samples might have been assigned to the wrong cluster.
    IN:data[n_samples,m_characteristics] = data to analyse
       max_n_clusters = Integer - Maximum number of clusters
       plot=boolean
    OUT: best_n_cluster = Integer - Number of clusters with best average silhouette score   
    """
    #The range of clusters is 2-max_n_clusters
    range_n_clusters = np.arange(2,max_n_clusters+1)
    range_silhouette_avg = np.zeros(max_n_clusters-1)
    
    for n_clusters in range_n_clusters: #loop over different number of clusters
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        range_silhouette_avg[n_clusters-2] = silhouette_avg
        
    if plot == True:
        plt.figure()
        plt.plot(range_n_clusters,range_silhouette_avg,'*-')
        plt.title(f'Cluster Analysis for data with {len(data[1]):d} characteristics',fontsize=18)
        plt.xlabel('Number of clusters',fontsize=16)
        plt.ylabel('Average Silhouette Number',fontsize=16)

def k_means_trafo(data,n_clusters,plot):
    """
    Computes the centers of K-Means clusters
    If m_characteristics == 3 the clusters can be visualized
    IN: data[n_samples,m_characteristics] = data to analyse
        n_clusters = Integer - Number of clusters to find in the given dataset
    OUT: centers[n_centers,m_characteristics] = cluster centers for the given data    
    """
    #init: initialization method ('k-means++','random'); n_clusters:Integer - number of cluster centroids;
    #n_init: Integer - number of random walks executed to optimize results 
    clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    cluster_labels = clusterer.fit_predict(data)
    centers = clusterer.cluster_centers_
    
    if (plot == True and len(data[0]) <=3 ):
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(data, cluster_labels)
        
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1) #Plot: Silhouette Score
        fig.set_size_inches(17, 7)   #Format size of subplots
        archColors = cm.plasma(np.arange(n_clusters).astype(float) / (n_clusters)) #shape: (n_clusters,4)
        dataColors = cm.plasma(cluster_labels.astype(float) / n_clusters)          #shape: (n_samples,4)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this eg. the range is -0.2 to 1
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            #color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values, facecolors=archColors[i,:], alpha=1)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        # Labeling for subplot 1: Silhouette scores
        ax1.set_title(f"Silhouette plot for various clusters. Silhouette Average = {silhouette_avg:.3f}"
                       ,fontsize=18)
        ax1.set_xlabel("Silhouette coefficient values",fontsize=16)
        ax1.set_ylabel("Cluster label",fontsize=16)
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        # 2nd Plot showing the actual clusters formed
        if (len(data[0]) == 2): 
            ax2 = fig.add_subplot(1,2,2)
            #Plot data
            ax2.scatter(data[:,0], data[:,1], c=dataColors, alpha = 0.6, s=5)
            #Plot cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],c=archColors,marker='X',edgecolor = 'k', alpha=1, s=200)
            ax2.set_title("K-Means clusteres of PCA reduced data.",fontsize=18)
            ax2.set_xlabel('1st pca component',fontsize=16)
            ax2.set_ylabel('2nd pca component',fontsize=16)
            plt.tight_layout()
            plt.show()
            
        elif len(data[0] == 3):
            ax2 = fig.add_subplot(1,2,2, projection='3d')
            #Plot data
            ax2.scatter(data[:,0], data[:,1], data[:,2], c=dataColors, alpha = 0.6, s=5)
            #Plot cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],centers[:, 2],c=archColors,marker='X',edgecolor = 'k', alpha=1, s=200)
            ax2.set_title("K-Means clusteres of PCA reduced data.",fontsize=18)
            ax2.set_xlabel('1st pca component',fontsize=16)
            ax2.set_ylabel('2nd pca component',fontsize=16)
            ax2.set_zlabel('3rd pca component',fontsize=16)
            plt.tight_layout()
            plt.show()
    return centers

#%% Outlier Analysis
def outlier_ana(data_scaled,mag_from_center):
    Center = np.mean(data_scaled,axis = 0)
    outliers = data_scaled >= mag_from_center*Center
    o_index = np.where(outliers== True)
    o_index = np.unique(o_index[0])
    data_scaled1 = np.delete(data_scaled,o_index,axis = 0)
    print(f'{len(o_index):d} samples are outliers and more than {mag_from_center:.1f} from centroid')
    return data_scaled1,Center
#data_scaled1 = outlier_ana(data_scaled,mag_from_center=40)

def blocking(tt,vector,blocks):
        """
        This is a function which helps to process big data files more easily
        by the method of block averaging. 
        For this the first argument is a vector with data, e.g. averaged temperature
        the second argument is another vector, e.g. time grid. 
        The third argument should be the number of blocks. 
        The more blocks, the more data points are taken into consideration. 
        If less blocks, more averaging takes place.
        """
        blockvec = np.zeros(blocks)
        elements = len(vector) 
        rest     = elements % blocks
        if rest != 0: #truncate vector if number of blocks dont fit in vector
            vector   = vector[0:-rest]
            tt       = tt[0:-rest]
            elements = len(vector)   
        meanA  = np.mean(vector)        
        bdata  = int((elements/blocks))#how many points per block
        sigBsq = 0; 
        for k in range(0,blocks):
            blockvec[k] = np.average(vector[k*bdata : (k+1)*bdata]) 
            sigBsq      = sigBsq + (blockvec[k]-meanA)**2    
        sigBsq *= 1/(blocks-1); 
        sigmaB = np.sqrt(sigBsq)
        error  = 1/np.sqrt(blocks)*sigmaB
        blocktt = tt[0:-1:bdata]
        return(blockvec,blocktt,error,sigmaB,bdata)

def blocked_feature_decay(decay,blocks,dists,gene_names,i_max,atype,featuretype,plot):
    """
    Block averaging of an input vector.
    IN:
     decay[2,n_samples] = array to block average (decay[0,:]=x-Axis, decay[1,:]=y-Axis)
     blocks = Number of blocks to block average the input vector
     For plotting:atype (0-noc) archetype to which the task decays
                  featuretype(0-m_tasks) decay of feature under consideration
    OUT:
     blockvec[blocks] = block averaged input vector y-Axis elements
     blocktt[blocks]  = x-Axis elements according to blockvec 
     error = statistical error in every block
     sigmaB
     bdata = number of elements compressed in every block
    """
    blockvec,blocktt,error,sigmaB,bdata = blocking(decay[0,:],decay[1,:],blocks=blocks)
    
    if (plot == True):
        n_a = np.shape(dists)[0]
        name = gene_names.iloc[i_max[atype,featuretype]]
        archColors = cm.plasma(np.arange(n_a).astype(float) / (n_a))
        plt.figure()
        plt.plot(blocktt,blockvec,'-*',c = archColors[atype,:])
        plt.suptitle(f"Blocked decay with respect to distance of A{atype}",fontsize = 14 )
        plt.title(f"{name}",fontsize=8)
        plt.xlabel(f"Normalized distance of samples to A{atype}",fontsize = 14)
        plt.ylabel(f"Amount in samples",fontsize = 14)
        plt.show()
    return blockvec,blocktt,error,sigmaB,bdata

def block_feature_decay(go_data,go_names,dists,min_index,blocks=10): 
    noc = np.shape(dists)[0]
    all_features1 = np.arange(0,np.shape(go_data)[1])
    all_features = []
    for k in range(0,noc): 
        all_features = np.append(all_features,all_features1)
    all_features = np.reshape(all_features,(noc,-1))
    all_features = all_features.astype(int)
    for i in range(0,len(min_index)):
        decay_i = feature_decay(go_data,go_names,
                dists,all_features,atype=i,featuretype=min_index[i],plot = False)
        blockvec,blocktt,error,sigmaB,bdata = blocked_feature_decay(decay_i,blocks,dists,go_names,
                                              all_features,atype=i,featuretype=min_index[i],plot=True)

def ranksum_test(decay,stbin_size,bdata,dists,atype,featuretype,plot):
    """
    Perfom a ranked sum test on input data decay. How likely is it that two sets
    of sample data origin from the same distribution. The decay of the first bin is 
    analyzed to the rest of the data.
    IN: decay[2,n_samples] = array to analyze: more precisely the decay[1,n_samoles] holds the 
        info about the decay and is analyzed.
        stbin_size(1-10): percent of information in the first bin in steps of 10%
        bdata: number of elements in one bin/block
        dists: distances to the archetype (input argument is passed from dist2A())
        atype(0-noc): archetype
        featuretype: featurty/task to analyze the distribution
    OUT: statistic   
         p_value: if small the 0Hypothesis (samples are drawn from the same distr.) is rejected.
    """
    
    stbin_data = decay[1,0:bdata*stbin_size]
    rest_data  = decay[1,bdata*stbin_size:]

    [statistic,p_value] = sp.stats.ranksums(stbin_data,rest_data) #sp.stats.ranksums(stbin_data,rest_data) 
    
    if plot == True:
        err_stbin = error*np.sqrt(stbin_size)
        err_rest  = error*np.sqrt(blocks-stbin_size)
        bottom = min([np.mean(stbin_data),np.mean(rest_data)])-max([err_stbin,err_rest])*5
    
        x = np.arange(2)
        n_a = np.shape(dists)[0]
        name = gene_names[i_max[atype,featuretype]]
        archColors = cm.plasma(np.arange(n_a).astype(float) / (n_a))
        
        plt.figure()
        plt.bar(x,[np.mean(stbin_data),np.mean(rest_data)]-bottom, color=archColors[atype,:],
                width=[0.1*stbin_size,0.1*(blocks-stbin_size)],bottom = bottom,
                yerr = [err_stbin,err_rest])
        plt.title(f'Mean decay of {name} with respect to distance of A{atype}',fontsize =18)
        plt.xticks(x,(f'First {stbin_size*10:d}% of samples','Rest'),fontsize=16)
        plt.ylabel(f"Average amount of {name} in samples",fontsize = 16)
        plt.tight_layout()
        plt.show()

    return statistic,p_value

def c_p_matrix(stbin_size,go_data,dists,noc,bdata,task_names):
    """
    Evaluates the coeff and p value of the decay slope w.r.t the distance to archetype
    IN: stbin_size(1-10): size of first bin - dataset compared with rest. H0: Data from same distr.
        go_data[n_samples,m_features] = data to analyze -> featurespace
        dists[archetype,n_sampes] = distancs form samplepoint to archetype in task/gene space
        noc = number of archetypes
        bdata = number of elements compressed in one blocking bin
    OUT: C_Matrix[noc,n_samples] = linear fit coeff for decay for archetypes and every task
         P_Matrix[noc,n_samples] = p_value for archetypes and samples (calculated with ranksums)
         c_min_index & p_min_index[noc,1] = indices for minimal slope/p_value indicate the task with
                                            maximal decay
    """
    all_features1 = np.arange(0,np.shape(go_data)[1]); all_features =[]
    for i in range(0,int(noc)): 
        all_features = np.append(all_features,all_features1,axis = 0)
    all_features = np.reshape(all_features,(noc,-1))
    all_features = all_features.astype(int)
    
    regressor = LinearRegression()
    C_Matrix = np.zeros((noc,len(go_data[1])))
    P_Matrix = np.zeros((noc,len(go_data[1])))
    for i in range(0,int(noc)): #loop over archetypes
        for j in range(0,len(go_data[1])): #loop over features/tasks 
            decay_j = feature_decay(go_data,task_names,dists,all_features,atype=i,featuretype=j,plot=False)
            blocks = 10 #number of blocks to compress data 
            i_max = np.zeros((noc,10)) #imax is arbetrary for plot=False
            blockvec,blocktt,_,_,_ = blocked_feature_decay(decay_j,blocks,dists,task_names,
                                                           i_max,atype=i,featuretype=j,plot=False) 
            
            #P value for decay in each feature t_i. Use bdata from blocked_feature
            statistic,p_value = ranksum_test(decay_j,stbin_size,bdata,dists,atype=j,featuretype=i,plot=False)
            P_Matrix[i,j] = p_value
            
            #fit a linear model y = ax + b to blocked data to find the minimum slope = coeff
            regressor.fit(np.array([blocktt]).T,np.array([blockvec]).T)
            coefficient = regressor.coef_  
            C_Matrix[i,j] = coefficient[0,0]
            
    p_min_index = np.argmin(P_Matrix,axis = 1)
    c_min_index = np.argmin(C_Matrix,axis = 1)
    return C_Matrix,c_min_index,P_Matrix,p_min_index,all_features

def show_minima(C_Matrix):
    """
    Show the first 3 most negative slopes of each arhcetype
    """
    sorted_c = np.sort(C_Matrix,axis = 1)
    if np.shape(C_Matrix)[0] == 4:
        labels = ['A0', 'A1', 'A2', 'A3']
    if np.shape(C_Matrix)[0] == 3:
        labels = ['A0', 'A1', 'A2']
    else: 
        print("This function only works for 3 or 4 Archetypes")
    c0 = -sorted_c[:,0]
    c1 = -sorted_c[:,1]
    c2 = -sorted_c[:,2]
    c3 = -sorted_c[:,4]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/0.8, c0, width/1.2, label=r'$-\lambda_0$')
    rects2 = ax.bar(x - width/2, c1, width/1.2, label=r'$-\lambda_1$')
    rects3 = ax.bar(x + width/7, c2, width/1.2, label=r'$-\lambda_2$')
    #rects4 = ax.bar(x + width/1, c3, width/1.2, label=r'$-\lambda_3$')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Negative slopes',fontsize = 14)
    ax.set_title('Comparing slopes of GO-expression decay',fontsize = 16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize = 14)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 0),  # 0 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    #autolabel(rects4)
    
    fig.tight_layout()
    plt.show()

def common_expressions(measureA,measureB,go_names,first_n):
    """
    common_expressions(c_indices_sorted,i_soted,gene_names,first_n = 50)
    """
    noc = np.shape(measureA)[0]
    for i in range(0,noc): 
        i_sorted_set = set(measureA[i,:first_n])
        c_min_set = set(measureB[i,:first_n])
        common = list(i_sorted_set & c_min_set)
        common = np.asanyarray(common)
        print(go_names.iloc[common])
        
        
        
def cip_listing(noc,go_names,C_Matrix,c_indices_sorted,A_go,i_sorted,
                P_Matrix,p_indices_sorted,save,filename):
    N = len(go_names)
    cip_list = np.chararray((N*noc+2,5),itemsize=60,offset=10,unicode=True)
    cip_list[0,4] = 'Feature_Expression'
    cip_list[0,3] = 'P-Value_(Ranked-Sum)'
    cip_list[0,2] = 'Slope_(Decay-coeff.)'
    cip_list[0,1] = 'Feature_Name_(GO)'
    cip_list[0,0] = 'Archetype_#'
    for i in range(0,noc): #loop over archetypes
        cip_list[(i*N)+1:((i+1)*N)+1,0] = f'A{i}' #GOName sorted by slope
        cip_list[(i*N)+1:((i+1)*N)+1,1] = go_names.iloc[c_indices_sorted[i,:]] 
        cip_list[(i*N)+1:((i+1)*N)+1,2] = C_Matrix[i,c_indices_sorted[i,:]]
        cip_list[(i*N)+1:((i+1)*N)+1,3] = P_Matrix[i,p_indices_sorted[i,:]]
        cip_list[(i*N)+1:((i+1)*N)+1,4] = A_go[i,i_sorted[i,:]]
        
    if (save == True):
        np.savetxt(filename,cip_list,fmt='%s')
    return cip_list        
