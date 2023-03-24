from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from mat2vec.processing import MaterialsTextProcessor
import sys 
sys.path.append('../')
import numpy as np 
import pandas as pd
from tqdm import tqdm
import re 


def get_wv_embedding(compositions):

	w2v_model = Word2Vec.load("../mat2vec/mat2vec/training/models/pretrained_embeddings")
	text_processor = MaterialsTextProcessor()

	save_dir = '../mat2vec_embedding/'

	for composition in tqdm(compositions):
		try:
			wv = w2v_model.wv[text_processor.process(composition)[0]]
			np.save(save_dir+'/{}.npy'.format(composition),wv)
		except:
			print(composition)

if __name__ == "__main__":

	mat_table = 'JMC_Data_Kappa_Unique_Averaged.csv'
	table = pd.read_csv(mat_table)
	#composition_name_new = sort_composition(table['composition_name'].values)

	get_wv_embedding(table['composition_name'].values)