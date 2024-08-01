# Progressive Non-Parametric Node Clustering

[Yue Liu](https://yliuhz.github.io/), [Zhongying Ru](https://www.linkedin.com/in/%E9%92%9F%E8%8E%B9-%E8%8C%B9-8b4732187/?locale=cs_CZ), Boyu Ruan


## Installation

On a Linux machine with Anaconda installed, run the following commands: 

```bash
git clone https://github.com/yliuhz/SAC && cd SAC 
conda create -n sac python=3.10.13
conda activate sac
python -m pip install -r requirements.txt
git clone https://github.com/yliuhz/networkit && cd networkit && python setup.py build_ext && python -m pip install -e . && cd -
python -m pip install .
```

## Get Started


```python
from sac import sac, sac_wo_hierarchy
import numpy as np
import networkit as nk

if __name__ == "__main__":
    ### Load pretrained node embeddings
    data = np.load("examples/GGD_amazon-photo_emb_0.npz")
    emb = data["emb"]

    ### Set the number of edges of the input graph
    m = 119081
    device = "cuda:0"


    ### Non-parametric node clustering
    preds = sac(emb, m, device=device) ## SAC
    preds = sac_wo_hierarchy(emb, m, device=device) ## SAC w/o hierarchical scaler


    preds, nkgraph = sac_wo_hierarchy(emb, m, device=device, return_graph=True) ## Check the similarity graph
    adj = nk.algebraic.adjacencyMatrix(nkgraph, matrixType='dense') ## Numpy.darray
    adj_sparse = nk.algebraic.adjacencyMatrix(nkgraph, matrixType='sparse') ## Scipy.sparse.csr_matrix
```