from matminer.featurizers.structure.sites import SiteStatsFingerprint
from sklearn.metrics.pairwise import cosine_similarity
from pymatgen.core import Structure
from glob import iglob
import warnings
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import pickle


def main():
	name_list = []
	vect_list = []

	feat = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")

	for names in iglob('./POSITIONS_2016/struct_comp_DB/DB/*/*'):
		print(names)
	#try:
		# read into pymatgen structure
		strc = Structure.from_file(names)

		# featurize the pymatgen structure
		feat_i = feat.featurize(strc)
		vect_list.append(feat_i)
		name_list.append(names.replace('/POSITIONS_2016/struct_comp_DB/DB/','').split('/')[-1])

		#except: 
		#	print('Failure:', names)

	vec_dic = {name_list[i]: vect_list[i] for i in range(len(name_list))}

	with open('featurized_vectors.pickle', 'wb') as f:
	    pickle.dump(vec_dic, f)


if __name__ == "__main__":
	main()