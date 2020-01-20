# Infearing-Key-Biological-Tasks
The key biological tasks of blood cancer (Acute Myeloid Leukemia) are investigated using an unsupervised machine learning model: A principal convex hull analysis (PCHA)[[1]] of the gene expressions of 451 patients revealed three main biological tasks for this type of cancer: cell proliferation, evading apoptosis, divert the immune system’s attentionaway from cancer cells.


Keywords: Pareto Optimization, Multi-Objective Optimization, Biological Trade-Offs, Archetype Analysis, PrincipalConvex Hull Analysis, Tumor Biology, Clustering 

### Work flow: 
1.  Preprocess Data
2.  Do PCA analysis
3.  Get rid of outliers
4.  Find Archetypes in Gene_space using PCHA
5.  Convert the Archetype coordinates from Gene- to Go- space
5.  Depict the spectrum of each archetype in Go-space.
6.  Decay analysis with slope and p- value




Main script: CIP_List.py

Methods Library: parti_lib.py

### Dependencies:
 [py_pchy](https://pypi.org/project/py-pcha/)
 [scikit-learn](https://scikit-learn.org/stable/)
 [Numpy](https://numpy.org/)
 [Scipy](https://www.scipy.org/)
 [Matplotlib](https://matplotlib.org/)
 [Pandas](https://pandas.pydata.org/)
 
 
 [1]:https://arxiv.org/abs/1901.10799
