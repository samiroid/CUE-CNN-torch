import subprocess
import shlex
import numpy as np
import random
import os
from itertools import product

base_dir = "/home/a.wrat/Sarcasm/sarcasm dataset/" 
ip = base_dir + "u2v_historical_data.txt"
emb = base_dir + "word_embeddings.txt"
op_base = "/home/a.wrat/Sarcasm/OUTPUT/u2v_trials_"

random.seed(42)
p = {
      'lr': [0.01, 0.1, 1, 10],
      'margin':[1,5,10],
      'min_word_freq': [2,3,4,5,6,7],
      'min_docs_user':[1,2,3,4,5],
      'neg_samples':[7,9,11,13]
      }

best_score = {'Train Accuracy': 0, 'Validation Accuracy': 0}
# function product takes cartesian product
# list of dicts
p_list = list((dict(zip(p.keys(), values)) for values in product(*p.values())))
random.shuffle(p_list)

reset = False
epochs = 20

for index, d in enumerate(p_list[:100]):
      lr = d['lr']
      margin = d['margin']
      min_word_freq = d['min_word_freq']
      min_docs_user = d['min_docs_user']
      neg_samples = d['neg_samples']
      encoding = "Latin-1"
      op = op_base+str(lr)+"_"+str(margin)+"_"+str(min_word_freq)+"_"+str(min_docs_user)+"_"+str(neg_samples)+"/"
      if os.path.exists(op+"U.txt"):
            continue
      subprocess.call(shlex.split('python -m user2vec.u2v -input "{}" -emb "{}" -output "{}" -emb_encoding {} -lr {} -margin {} -min_word_freq {} -min_docs_user {} -neg_samples {} -epochs {}'.format(ip, emb, op, encoding, lr, margin, min_word_freq, min_docs_user, neg_samples, epochs)))