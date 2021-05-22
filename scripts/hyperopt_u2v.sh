#!/bin/bash
base_dir="/home/a.wrat/"
OP=$base_dir'results.txt'
wdir=$base_dir'Sarcasm/sarcasm dataset/word_embeddings.txt'
u2vdir='/home/a.wrat/Sarcasm/OUTPUT/*'
dataset=$base_dir'Sarcasm/sarcasm dataset/experiment.csv'
python cue_cnn/cnn/hyperopt_u2v.py -output "$OP" -data "$dataset" -pretrained-embed-words True -emb-words "$wdir" -emb-users-root "$u2vdir"