import torch
import sys
sys.path.append('../MatSciBERT')
from normalize_text import normalize
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import re
from tqdm import tqdm
import os
import numpy as np

def sort_composition(compositions):

	composition_name_new = []
	
	for name in compositions:
		composition = name
		composition_split = re.findall('[A-Z][^A-Z]*', composition)
		composition_split.sort()

		for i, element in enumerate(composition_split):
			if not element[-1].isnumeric():
				composition_split[i] = composition_split[i] + '1'
		composition_sorted = "".join(composition_split)
		composition_name_new.append(composition_sorted)

	return composition_name_new


def make_robodata(save_dir, robo_data = None):

	if robo_data is None:
		robo_dir = 'robo_descriptions'

		file1 = open(robo_dir, "r")
		lines = file1.readlines()
		file1.close()

		lines = [l[:-1] for l in lines]
		names = [l.split(' ')[0] for l in lines]
		
		robo_data = pd.DataFrame({'composition_name':names,'sentence':lines})

	mat_table = 'JMC_Data_Kappa_Unique_Averaged.csv'
	table = pd.read_csv(mat_table)
	names_new = sort_composition(robo_data['composition_name'].values)	
	robo_data['composition_name'] = names_new
	composition_name_new = sort_composition(table['composition_name'].values)

	robo_data_use = robo_data[(np.in1d(names_new, composition_name_new))]

	robo_data_use.to_pickle(save_dir+'/robo_data_kappa.pkl')

	return robo_data_use


def run_matscibert(robo_data):

	save_dir = '../robo_matscibert_embedding'
	model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
	tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
	#sentences = ['SiO2 is a network former.']
	embeddings = []
	for row in tqdm(robo_data.iterrows()):

		sentence = row[1].sentence
		composition = row[1].composition_name

		norm_sents = [normalize(s) for s in [sentence]]
		tokenized_sents = tokenizer(norm_sents)
		tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}
		token_size = int(tokenized_sents['input_ids'].shape[1])
		sentence_embedding = []
		for i in range(0,token_size,512):
			with torch.no_grad():
				last_hidden_state = model(input_ids=tokenized_sents['input_ids'][:,i:i+512],token_type_ids=tokenized_sents['token_type_ids'][:,i:i+512],
				attention_mask=tokenized_sents['attention_mask'][:,i:i+512])[0]

			sentence_embedding.append(last_hidden_state.detach().cpu().numpy()[0])
			#np.save(save_dir+'/{}.npy'.format(composition),last_hidden_state.detach().cpu().numpy()[0])
			
		embeddings.append(np.concatenate(sentence_embedding).mean(0).reshape(1,-1))

	robo_data['embeddings'] = embeddings
	robo_data.to_pickle('robo_descriptions_with_embed_noprop.pkl')

if __name__ == "__main__":

	save_dir = '../'
	data = pd.read_pickle('robo_descriptions.pkl')

	if not os.path.isfile(save_dir+'robo_data_kappa_with_prop.pkl'):
		robo_data = make_robodata(save_dir, data)
	else:
		robo_data = pd.read_pickle(save_dir+'robo_data_kappa_with_prop.pkl')

	run_matscibert(data)