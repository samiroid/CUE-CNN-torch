#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
import torchtext.datasets as datasets
import torchtext.vocab as vocab
import model
import train
import mydatasets
import pandas as pd
import random
from dataloader import dataloader


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=40, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=200, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=100, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=False, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=0.01, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=60, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5,6,7', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-data', type=str, default='Sarcasm/sarcasm dataset/experiment.csv', help='Dataset Directory')
parser.add_argument('-pretrained-embed-words', type=bool, default=False, help='Use pre-trained embedding for words [default: False]')
parser.add_argument('-pretrained-embed-users', type=bool, default=False, help='Use pre-trained embedding for users [default: False]')
parser.add_argument('-emb-words', type=str, default='Sarcasm/sarcasm dataset/word_embeddings.txt' , help='Directory of pretrained word embeddings')
parser.add_argument('-emb-users', type=str, default='sarcasm dataset/user_embeddings.txt', help='Directory of pretrained user embeddings')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# hyper param learning, if true, outputs only best test acc
parser.add_argument('-param-search', type=bool, default=False, help='Tuning hyper parameters')
parser.add_argument('-conv-layer', type=int, default=100, help='Hidden layer of the last convolution layer')
args = parser.parse_args()

# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
user_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = dataloader(text_field,label_field,user_field,args,wdir=args.emb_words,u2vdir=args.emb_users,device='cpu',repeat=False)

# update args and print
words_vec = text_field.vocab.vectors
users_vec = user_field.vocab.vectors
# print(user_field.vocab.value)
#print(torch.sum(torch.sum(users_vec,1)!=0))
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.embed_num_users = len(user_field.vocab)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))

# model
args.embed_dim_users = 100
cnn = model.CNN_Text(args.embed_num_users, args.embed_num, args.embed_dim_users, 
                      args.embed_dim, args.class_num, args.kernel_num, args.kernel_sizes, 
                      args.conv_layer, args.dropout, args.pretrained_embed_words, 
                      args.pretrained_embed_users, words_vec, users_vec)
                                                        
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args.cuda, args.epochs, args.lr, args.max_norm, args.log_interval, args.test_interval, args.save_best, args.early_stop, args.save_interval, args.save_dir)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

