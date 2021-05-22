#!/bin/bash
base_dir="/home/a.wrat/"
wdir=$base_dir'Sarcasm/sarcasm dataset/word_embeddings.txt'
u2vdir="/home/a.wrat/Sarcasm/OUTPUT/u2v_trials_10_10_3_4_9/U.txt"
dataset=$base_dir'Sarcasm/sarcasm dataset/experiment.csv'
python cue_cnn/cnn/main.py -data "$dataset" -pretrained-embed-words True -pretrained-embed-users True -emb-words "$wdir" -emb-users "$u2vdir"