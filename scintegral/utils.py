import numpy as np
import pandas as pd

"""
	Utils
	~~~~~~


"""


def marker_input_creator(marker_dict):
	"""
	Converts a list of markers to a marker-onehot format.
	

	:param dict marker_dict: A dictionary where keys are cell-types and values are list of markers.


	:returns DataFrame: A marker-onehot format used as an input for the classifier.

	"""
	marker_onehot = pd.DataFrame(
			index=sum(list(marker_dict.values()),[]),
			columns=marker_dict.keys())
	
	for key, value in marker_dict.items():
		marker_onehot.loc[value,key] = 1

	marker_onehot.fillna(0, inplace=True)
	marker_onehot['others'] = 0

	return marker_onehot

