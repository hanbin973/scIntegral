# Python pacakge: scIntegral

[![Documentation Status](https://readthedocs.org/projects/scintegral/badge/?version=latest)](https://scintegral.readthedocs.io/en/latest/?badge=latest) 

## Installation

`scintegral` can be installed using pip
```
pip install scintegral
```

## Basic Usage

scIntegral's cell classifier can be loaded through
```python
import scintegral.classifer as scint.classifier
import scintegral.utils as scint.utils
```

scIntegral requires a marker information for each cell-type.
Given a python dictionary of the following format,
```python
marker_dict = {
	...
	'Fibroblasts':{'Col3a1', 'Col8a1'}
	...
}
```
 run 
```python
scint.utils.marker_input_creator(marker_dict)
```
which returns a onehot `pandas` dataframe in which scIntegral takes as an input.

Finally, to run scIntegral's cell classifier,
```python
scint.classifier.classify_cells(...)
```

The arguments can be found at the [documentation](https://scintegral.readthedocs.io/en/latest/index.html) page.




