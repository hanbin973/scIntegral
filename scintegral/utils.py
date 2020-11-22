import numpy as np
import pandas as pd

def marker_input_creator(marker_dict):
	marker_onehot = pd.DataFrame(
			index=sum(list(marker_dict.values()),[]),
			columns=marker_dict.keys())
	
	for key, value in marker_dict.items():
		marker_onehot.loc[value,key] = 1

	marker_onehot.fillna(0, inplace=True)
	marker_onehot['others'] = 0

	return marker_onehot

