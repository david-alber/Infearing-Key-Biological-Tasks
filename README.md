# Infearing-Key-Biological-Tasks
The key biological tasks of blood cancer (Acute Myeloid Leukemia) are investigated using an unsupervised machine learning model: A principal convex hull analysis (PCHA)[[1]] of the gene expressions of 451 patients revealed three main archetypes (A0, A1, A2) and hence three main biological specifications for this type of cancer: divert the immune system’s attentionaway from cancer cells (A0), cell proliferation (A1), evading apoptosis (A2).


Keywords: Pareto Optimization, Multi-Objective Optimization, Biological Trade-Offs, Archetype Analysis, PrincipalConvex Hull Analysis, Tumor Biology, Clustering 

### Work flow: 
1.  Preprocess Data
2.  Do PCA analysis
3.  Get rid of outliers
4.  Find Archetypes in Gene_space using PCHA
5.  Convert the Archetype coordinates from Gene- to Go- space
5.  Depict the spectrum of each archetype in Go-space.
6.  Decay analysis with slope and p- value


### Archetype Analysis
 Archetype Depiction 
:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/37decf4c2387de070ba6d8f462c58bfb2a01a366/Images/archePlot.png" width="320" height="300" />  
 
Projection of the high dimensional gene- expression tumorsamples onto the first two principle components. The three vertexpoints/ archetypes highlighted correspond to distinct features andspan the triangle of the pareto front, such that all points within canbe explained as convex combinations of the vertices.

### Biological Interpretation
  A0 | A1 | A2  
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" />  |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" /> |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A2decay.png" width="320" height="300" />

Selection of one GO- expression for each archetype, according to the maximal descent away from the archetype, i.e. most negativeslope for linear regression.  Suggesting thatBiocarta Blymphocyte Phathway,Reactome Unwinding of DNAandReactome EndosomalVacuolary Pathwaycan be linked to key biological tasks.



### Dependencies:
 [py_pchy](https://pypi.org/project/py-pcha/), 
 [scikit-learn](https://scikit-learn.org/stable/), 
 [Numpy](https://numpy.org/), 
 [Scipy](https://www.scipy.org/), 
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/)
 
 
 [1]:https://arxiv.org/abs/1901.10799
