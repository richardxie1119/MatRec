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
	vec_dict = {}
	feat = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")
	fail_names = []
	names_all = list(iglob('./POSITIONS_2016/struct_comp_DB/DB/*/*'))
	save_dir = '../fingerprints'

	for names in tqdm(names_all):

		try:
			# read into pymatgen structure
			strc = Structure.from_file(names)

			# featurize the pymatgen structure
			feat_i = feat.featurize(strc)
			#vect_list.append(feat_i)
			name_ = names.replace('/POSITIONS_2016/struct_comp_DB/DB/','').split('/')[-1]

			#name_list.append(names.replace('/POSITIONS_2016/struct_comp_DB/DB/','').split('/')[-1])
			vec_dict[name_] = feat_i

			with open(save_dir+'/{}.pkl'.format(name_), 'wb') as f:
	    		pickle.dump(feat_i, f)

		except: 
			#print('Failure:', names)
			fail_names.append(names)

	#vec_dict = {name_list[i]: vect_list[i] for i in range(len(name_list))}

	# with open('featurized_vectors.pickle', 'wb') as f:
	#     pickle.dump(vec_dict, f)

	fail_names = pd.DataFrame(fail_names)
	fail_names.to_csv(save_dir+'/fail_names.csv')

if __name__ == "__main__":
	main()