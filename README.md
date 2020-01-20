# Infearing-Key-Biological-Tasks
The key biological tasks of blood cancer (Acute Myeloid Leukemia) are investigated and analyzing with a multi objective optimization approach introduced in the framework of Archetypal Analysis (AA) by [[2]] and is current subject to research as in [[3]] and [[4]]. Here an unsupervised machine learning model, a principal convex hull analysis (PCHA)[[1]], is used to reveal the three main biological specifications for this type of cancer: divert the immune system’s attentionaway from cancer cells (A0), cell proliferation (A1), evading apoptosis (A2).


Keywords: Pareto Optimization, Multi-Objective Optimization, Biological Trade-Offs, Archetype Analysis, PrincipalConvex Hull Analysis, Tumor Biology, Clustering 

### Work flow: 
1.  Preprocess Data
2.  Do PCA analysis
3.  Get rid of outliers
4.  Find Archetypes in Gene_space using PCHA
5.  Convert the Archetype coordinates from Gene- to Go- space
5.  Depict the spectrum of each archetype in Go-space.
6.  Conclude biological task: Decay analysis with slope and p- value

### Getting started
GoMatrix is found in Data -> c2.cp.v4.0.symbols.gmt 

```python
import numpy as np
import pandas as pd 
from parti_lib import *
from py_pcha import PCHA

#%% Load data
gene_data = pd.read_csv("FilepathToData/GeneData.txt",delimiter= "\t")
gene_data = gene_data.transpose()

names = np.arange(0,160) #place holders to seperate gene_expr in GoData-Matrix
GoData = pd.read_csv("FilepathToGoMatrix/GoMatrix.gmt",
                     delimiter="\t", names = names) 
GoData = GoData.drop(columns = 1,axis = 1)
```

### Archetype Analysis
 Archetype Depiction 
:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/37decf4c2387de070ba6d8f462c58bfb2a01a366/Images/archePlot.png" width="320" height="300" />  
 
Projection of the high dimensional gene- expression tumorsamples onto the first two principle components. The three vertexpoints/ archetypes highlighted correspond to distinct features andspan the triangle of the pareto front, such that all points within canbe explained as convex combinations of the vertices.

### Biological Interpretation
  A0 | A1 | A2  
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" />  |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" /> |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A2decay.png" width="320" height="300" />

Selection of one GO- expression for each archetype, according to the maximal descent away from the archetype, i.e. most negative slope for linear regression.  Suggesting that Biocarta Blymphocyte Phathway, Reactome Unwinding of DNA and Reactome Endosomal Vacuolary Pathwaycan be linked to key biological tasks.

### How to contribute
Fork from the `Developer`- branch and pull request to merge back into the original `Developer`- branch. 
Working updates and improvements will then be merged into the `Master` branch, which will always contain the latest working version.

With: 
* [Lukas Alber](https://github.com/luksen99)
* [Jean Hausser](https://www.scilifelab.se/researchers/jean-hausser/)

### Dependencies
 [py_pchy](https://pypi.org/project/py-pcha/), 
 [scikit-learn](https://scikit-learn.org/stable/), 
 [Numpy](https://numpy.org/), 
 [Scipy](https://www.scipy.org/), 
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/)
 
 
 [1]:https://arxiv.org/abs/1901.10799
 [2]:https://www.tandfonline.com/doi/abs/10.1080/00401706.1994.10485840
 [3]:https://science.sciencemag.org/content/336/6085/1157/tab-article-info
 [4]:https://www.nature.com/articles/nmeth.3254
 [4]: 
