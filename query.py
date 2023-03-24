import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import os
import scipy.stats as ss
from tqdm import tqdm

data_dir = '../robo_matscibert_embedding/'

def query_composition(query, compositions):
    
    """_summary_

    Args:
        query (_type_): _description_
        compositions (_type_): _description_

    Returns:
        _type_: _description_
    """

    query_embedding = np.load(data_dir+'{}.npy'.format(query))
    query_embedding = query_embedding.mean(0).reshape(1,-1)
    
    composition_embeddings = []
    composition_names = []

    for composition in compositions:
        
        if os.path.isfile(data_dir+'{}.npy'.format(composition)):
            embedding = np.load(data_dir+'{}.npy'.format(composition))
            embedding = embedding.mean(0).reshape(1,-1)
            composition_embeddings.append(embedding)
            composition_names.append(composition)
            
    cos_sim = cosine_similarity(query_embedding,np.concatenate(composition_embeddings))
    
    rank = sorted(list(cos_sim[0]))[::-1]
    index = [rank.index(v) for v in list(cos_sim[0])]
    
    return pd.DataFrame({'similarity':cos_sim[0],'rank':index,'composition_name':composition_names})


def eval_query():

    targets = ['beta (p)','beta (n)']
    db = pd.read_csv('Database – TEDesignLab.csv')

    rank_target = []
    rank_query = []
    labels = []
    query_compositions = []

    for i in tqdm(range(len(db['composition_name']))):
        composition = db['composition_name'].iloc[i]

        if os.path.isfile(data_dir+'{}.npy'.format(composition)):
            query_compositions.append(composition)
            query_df = query_composition(db['composition_name'].iloc[i],db['composition_name'].values)
            rank_query.append(query_df.iloc[1:])
            db_query = db[np.in1d(db['composition_name'],query_df['composition_name'])]
            labels = []
            label_values = []

            for target in targets:
                label = abs(db_query[db_query['composition_name']==composition][target].values-
                            db_query[db_query['composition_name']!=composition][target].values)
                labels.append(label/db_query[db_query['composition_name']==composition][target].values)
                label_values.append(label)
            labels = np.array(labels).sum(0)
            rank = sorted(list(labels))
            index = [rank.index(v) for v in list(labels)]
            rank_target.append(pd.DataFrame({'label':labels,'rank':index,'label1':label_values[0],'label2':label_values[1],
                                            'composition_name':db_query[db_query['composition_name']!=composition]['composition_name']}))