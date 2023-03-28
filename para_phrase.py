
from parrot import Parrot
import torch
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import pandas as pd 


''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)

import torch.nn as nn

class Parrot():
  
  def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    import pandas as pd
    from parrot.filters import Adequacy
    from parrot.filters import Fluency
    from parrot.filters import Diversity
    self.tokenizer = AutoTokenizer.from_pretrained(model_tag, use_auth_token=False)
    self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_tag, use_auth_token=False)
    self.adequacy_score = Adequacy()
    self.fluency_score  = Fluency()
    self.diversity_score= Diversity()

  def rephrase(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False, style=1, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      if use_gpu:
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
      else:
        device = "cpu"
      self.model= nn.DataParallel(self.model)
      self.model     = self.model.to(device)
      import re
      save_phrase = input_phrase
      if len(input_phrase) >= max_length:
         max_length += 32 	
      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(device)
      max_return_phrases = 10
      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break 
        #print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)            
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        
      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)

         

      adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device )
      if len(adequacy_filtered_phrases) > 0 :
        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device )
        if len(fluency_filtered_phrases) > 0 :
            diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
            para_phrases = []
            for para_phrase, diversity_score in diversity_scored_phrases.items():
                para_phrases.append((para_phrase, diversity_score))
            para_phrases.sort(key=lambda x:x[1], reverse=True)
            return para_phrases[0]
        else:
            return [(save_phrase,0)]

  def augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False, max_return_phrases = 10, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      if use_gpu:
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
      else:
        device = "cpu"
      self.model= nn.DataParallel(self.model)
      self.model     = self.model.to(device)

      import re

      save_phrase = input_phrase
      if len(input_phrase) >= max_length:
         max_length += 32	
			
      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(device)

      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break 
        #print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)            
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        

      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)


      adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device )
      if len(adequacy_filtered_phrases) > 0 :
        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device )
        if len(fluency_filtered_phrases) > 0 :
            diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
            para_phrases = []
            for para_phrase, diversity_score in diversity_scored_phrases.items():
                para_phrases.append((para_phrase, diversity_score))
            para_phrases.sort(key=lambda x:x[1], reverse=True)
            return para_phrases
        else:
            return [(save_phrase,0)]


def get_para_phrase(model, data):

	for i in tqdm(range(data.shape[0])):

		sentence = data.sentence.iloc[i]
		phrases = sentence.split('. ')

		sentence_para = []
		for phrase in phrases:
			try:
				para_phrases = model.augment(input_phrase=phrase,
										use_gpu=True,
										do_diverse=False,             # Enable this to get more diverse paraphrases
										adequacy_threshold = 0.50,   # Lower this numbers if no paraphrases returned
										fluency_threshold = 0.50,
										max_return_phrases = 5)
				sentence_para.append(para_phrases)
			except:
				sentence_para.append([])
		
		data['sentence_para'].iloc[i] = sentence_para
	
	return data

if __name__ == "__main__":

    data = pd.read_pickle('robo_descriptions.pkl')
    data['sentence_para'] = data['sentence'].copy()
    data['sentence_para'] = data['sentence_para'].astype(object)

    model = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    para_data = get_para_phrase(model, data)
    para_data.to_pickle('robo_descriptions_para.pkl')