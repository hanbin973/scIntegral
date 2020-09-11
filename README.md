# Scalable integration and classification of single cell RNA sequencing data

This is the github repository of scIntegral (Lee et al.).
We provide a brief tutorial of how to use the software using the open Koh et al. dataset.

## scIntegral tutorial using human embryogenic stem cell data (Koh et al.)

### Load libraries
In python, data are frequently stored in two forms : numpy arrays and pandas dataframes.
We first load the data using these packages and convert them to torch tensor formats afterwards since scIntegral uses PyTorch as a backend.
For visualization, matplotlib and seaborn is going to be used.
The following codes loads the required packages.


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from scipy.io import mmread
```

### Torch configurations
When using multiple gpu devices, the gpu being used must be specified.
Now we store these devices in variables for future use.
Additionally, in both CPU and GPU, storing data in float32 demands less memory and cpu resource.
Therefore, we use float32 instead of float64 (double) as the default datatype.


```python
dtype = torch.float32
torch.set_default_dtype(dtype)
device_cpu = torch.device('cpu')
device_cuda_list = [torch.device("cuda:{}".format(i)) for i in range(6)[::-1]]
```

### Load data

Here we load three files : the expression count matrix (koh.data.counts.mm), row names (=genes, koh.data.row) and column names (=samples, koh.data.col).


```python
exp_data=mmread('koh_extract/koh.data.counts.mm').toarray().astype(float)
with open('koh_extract/koh.data.col','r') as f: exp_data_col=[i.strip().strip('"') for i in f.read().split()]
with open('koh_extract/koh.data.row','r') as f: exp_data_row=[i.strip().strip('"') for i in f.read().split()]
assert exp_data.shape==(len(exp_data_row),len(exp_data_col))
```

### Load meta data

This metadata file contians the true cell type labels in the 'celltype' column.
Precomputed tSNE coordinates are also contained in this file.


```python
exp_data_meta=pd.read_csv('koh_extract/koh.metadata.tsv',sep='\t')
exp_data_meta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Run</th>
      <th>LibraryName</th>
      <th>phenoid</th>
      <th>libsize.drop</th>
      <th>feature.drop</th>
      <th>total_features</th>
      <th>log10_total_features</th>
      <th>total_counts</th>
      <th>log10_total_counts</th>
      <th>pct_counts_top_50_features</th>
      <th>pct_counts_top_100_features</th>
      <th>pct_counts_top_200_features</th>
      <th>pct_counts_top_500_features</th>
      <th>is_cell_control</th>
      <th>celltype</th>
      <th>tSNE_1</th>
      <th>tSNE_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SRR3952323</th>
      <td>SRR3952323</td>
      <td>H7hESC</td>
      <td>H7hESC</td>
      <td>False</td>
      <td>False</td>
      <td>4895</td>
      <td>3.689841</td>
      <td>2.248411e+06</td>
      <td>6.351876</td>
      <td>18.278965</td>
      <td>25.975390</td>
      <td>35.537616</td>
      <td>52.410941</td>
      <td>False</td>
      <td>hESC</td>
      <td>9.973465</td>
      <td>19.045918</td>
    </tr>
    <tr>
      <th>SRR3952325</th>
      <td>SRR3952325</td>
      <td>H7hESC</td>
      <td>H7hESC</td>
      <td>False</td>
      <td>False</td>
      <td>4887</td>
      <td>3.689131</td>
      <td>2.271617e+06</td>
      <td>6.356335</td>
      <td>24.672529</td>
      <td>32.222803</td>
      <td>41.547358</td>
      <td>57.969233</td>
      <td>False</td>
      <td>hESC</td>
      <td>10.366232</td>
      <td>21.511833</td>
    </tr>
    <tr>
      <th>SRR3952326</th>
      <td>SRR3952326</td>
      <td>H7hESC</td>
      <td>H7hESC</td>
      <td>False</td>
      <td>False</td>
      <td>4888</td>
      <td>3.689220</td>
      <td>5.846824e+05</td>
      <td>5.766921</td>
      <td>22.732839</td>
      <td>30.205988</td>
      <td>39.431308</td>
      <td>55.285817</td>
      <td>False</td>
      <td>hESC</td>
      <td>9.881356</td>
      <td>19.317197</td>
    </tr>
    <tr>
      <th>SRR3952327</th>
      <td>SRR3952327</td>
      <td>H7hESC</td>
      <td>H7hESC</td>
      <td>False</td>
      <td>False</td>
      <td>4879</td>
      <td>3.688420</td>
      <td>3.191810e+06</td>
      <td>6.504037</td>
      <td>20.867378</td>
      <td>29.003904</td>
      <td>38.785558</td>
      <td>56.020859</td>
      <td>False</td>
      <td>hESC</td>
      <td>8.483966</td>
      <td>21.289459</td>
    </tr>
    <tr>
      <th>SRR3952328</th>
      <td>SRR3952328</td>
      <td>H7hESC</td>
      <td>H7hESC</td>
      <td>False</td>
      <td>False</td>
      <td>4873</td>
      <td>3.687886</td>
      <td>2.190385e+06</td>
      <td>6.340521</td>
      <td>21.287923</td>
      <td>29.423689</td>
      <td>39.307683</td>
      <td>56.640975</td>
      <td>False</td>
      <td>hESC</td>
      <td>9.017168</td>
      <td>20.637262</td>
    </tr>
  </tbody>
</table>
</div>



### Marker Info
This file contains the marker specification constructed based on the procedure described in the manuscript.


```python
clustername_to_markers=pd.read_csv('/data01/ch6845/dynamic_cell_classifier/data/koh_extract/koh.rho.tsv',sep='\t').T
```

### Preprocessing

Here we subset the used genes, construct covariate matrix X.


```python
marker_unique=list(clustername_to_markers.columns)
marker_unique_exp_data_idx=[exp_data_row.index(marker) for marker in marker_unique]

cell_size_factor=pd.read_csv('koh_extract/koh.size_factor_cluster.tsv',sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)
cell_size_factor.shape

Y=exp_data[marker_unique_exp_data_idx].transpose().astype(float)
Y.shape

marker_onehot=clustername_to_markers

# np.zeros in first line for non-baseline model, np.ones for baseline model
x_data_intercept=np.array([np.ones(Y.shape[0])]).transpose()
x_data_null=np.concatenate([x_data_intercept],axis=1)
x_data_null.shape
```




    (446, 1)




```python
# review the variables before using them
Y
s = cell_size_factor
X = x_data_null.copy()
rho = marker_onehot.T
```

# Run scIntegral (GPU-mode)


### Load scintegral


```python
import sys
sys.path.insert(0, "../")
import scintegral as scint
```

### Conversion to Torch format
Now, we convert all the input variables to the PyTorch format before use.


```python
# when using gpu, you must specify device (gpu) number that is being used
with torch.cuda.device(4):
    Y = torch.tensor(Y, dtype=dtype).cuda()
    X = torch.tensor(X, dtype=dtype).cuda()
    s = torch.tensor(s, dtype=dtype).cuda()
```

The device number when calling scIntegral (4 in this tutorial) must match with device number in which the data was loaded 


```python
with torch.cuda.device(4):
    cell_types, post_probs = scint.classify(Y, X, s, 8, rho, use_gpu=True)
```

    Loss at initialization: 254943.219
    Start optimization
    0-th iteration, Loss: 250433.953
    20-th iteration, Loss: 211579.125
    40-th iteration, Loss: 209208.047


We plot the classification result in tSNE using seaborn


```python
sns.scatterplot('tSNE_1', 'tSNE_2', hue=cell_types, data=exp_data_meta)
plt.show()
```


![png](output_24_0.png)

