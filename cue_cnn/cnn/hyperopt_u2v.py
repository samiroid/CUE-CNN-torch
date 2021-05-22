b_dir = '/home/a.wrat/'

import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
# import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
from dataloader import dataloader
import pandas as pd
import random
import glob

from itertools import product
import numpy as np

random.seed(42)

parser = argparse.ArgumentParser(description='User2Vec (CNN) hyper parameter tuning')
parser.add_argument('-output', type=str, default="output.txt", help='The directory for saving the results [default: output.txt]')
parser.add_argument('-data', type=str, default='Sarcasm/sarcasm dataset/experiment.csv', help='Dataset Directory')
parser.add_argument('-pretrained-embed-words', type=bool, default=False, help='Use pre-trained embedding for words [default: False]')
parser.add_argument('-pretrained-embed-users', type=bool, default=True, help='Use pre-trained embedding for users [default: True]')
parser.add_argument('-emb-words', type=str, default='Sarcasm/sarcasm dataset/word_embeddings.txt' , help='Directory of pretrained word embeddings')
parser.add_argument('-emb-users-root', type=str, default='Sarcasm/OUTPUT/', help='Directory of pretrained user embeddings')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# hyper param learning, if true, outputs only best test acc
parser.add_argument('-param-search', type=bool, default=True, help='Tuning hyper parameters')
args = parser.parse_args()

hyp = {
      'lr': 0.001,
      'dropout':0.5, 
      'max_norm':0.01,
      'kernel_num':80,
      'kernel_size':'5,6,7',
      'conv_layer':150
      }

d = args.emb_users_root
p_list = glob.glob(d)
print(p_list)

best_score = {'Train Accuracy': 0, 'Validation Accuracy': 0}
with open(args.output, 'w') as filehandle:
    filehandle.writelines("CNN with pretrained user embeddings \n")
    # filehandle.writelines("%s " % attr for attr,_ in best_score.items())
    filehandle.writelines("Tr_acc ")
    filehandle.writelines("Val_acc ")
    filehandle.writelines("lr margin min_word_freq min_docs_user neg_samples")    
    f_best = p_list[0]
    for fname in p_list:        
        # learning
        lr = float(hyp['lr'])
        epochs = 10
        log_interval = 1
        test_interval = 20
        save_interval = 500
        save_dir = 'snapshot'
        early_stop = 100
        save_best = False

        # model
        dropout = float(hyp['dropout'])
        max_norm = float(hyp['max_norm'])
        embed_dim = 128
        embed_dim_users = 128
        kernel_num = int(hyp['kernel_num'])
        kernel_sizes = hyp['kernel_size']
        conv_layer = int(hyp['conv_layer'])
        # device
        # data 


        # load data
        # print("\nLoading data...")
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        user_field = data.Field()
        train_iter, dev_iter, test_iter = dataloader(text_field, label_field, user_field, args, wdir=args.emb_words, u2vdir=fname+'/U.txt', device='cpu', repeat=False)
        
        # update args and print
        words_vec = text_field.vocab.vectors
        users_vec = user_field.vocab.vectors
        # print(users_vec)
        emb_num = len(text_field.vocab)
        class_num = len(label_field.vocab) - 1
        emb_num_u = len(user_field.vocab)
        cuda = (not args.no_cuda) and torch.cuda.is_available()
        ker_siz = [int(k) for k in kernel_sizes.split(',')]
        sav_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # print("\nParameters:")
        # for attr, value in sorted(args.__dict__.items()):
        #     print("\t{}={}".format(attr.upper(), value))

        # model
        cnn = model.CNN_Text(emb_num_u, emb_num, embed_dim_users, embed_dim, class_num, 
                kernel_num, ker_siz, conv_layer, dropout, args.pretrained_embed_words, 
                args.pretrained_embed_users, words_vec, users_vec)

        if args.snapshot is not None:
            print('\nLoading model from {}...'.format(args.snapshot))
            cnn.load_state_dict(torch.load(args.snapshot))

        if cuda:
            torch.cuda.set_device(args.device)
            cnn = cnn.cuda()

        # train or predict
        if args.predict is not None:
            label = train.predict(args.predict, cnn, text_field, label_field, user_field, args.cuda)
            print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
        elif args.test:
            try:
                train.eval(test_iter, cnn, args) 
            except Exception as e:
                print("\nSorry. The test dataset doesn't  exist.\n")
        else:
            print()
            try:
                tr_acc, dev_acc, last_step = train.train(train_iter, dev_iter, cnn, cuda, epochs, lr, max_norm, 
                            log_interval, test_interval, save_best, early_stop, 
                            save_interval, save_dir)
            except KeyboardInterrupt:
                break
                print('\n' + '-' * 89)
                print('Exiting from training early')
        
        if args.param_search:
            print("\n Performance of the Model with Params defined below",'\n',
            "Train Accuracy: {:.2f} % \n Validation Accuracy: {:.2f}% \n Steps: {:.0f}".format(tr_acc,dev_acc,last_step))
            print("\n Hyper Parameters:")
            for attr, value in sorted(hyp.items()):
                print("\t{} = {}".format(attr.upper(), value))

            print('\n ############################################################ \n')
            if dev_acc > best_score['Validation Accuracy']:
                test_acc = train.eval(test_iter, cnn, args) 
                best_score['Train Accuracy'] = tr_acc
                best_score['Validation Accuracy'] = dev_acc
                f_best = fname
                # p_best = hyp.copy()

        filehandle.writelines("\n")
        filehandle.writelines("%s " % round(float(tr_acc),2))
        filehandle.writelines("%s " % round(float(dev_acc),2))
        filehandle.writelines("%s " % val for val in fname.split('/')[-1].split('_')[2:])
        
    filehandle.writelines('\n\n' + '-' * 89)
    filehandle.writelines("\n Best set of hyper parameters\n")
    filehandle.writelines("lr margin min_word_freq min_docs_user neg_samples")    
    filehandle.writelines("\n")
    filehandle.writelines(" %s"% val for val in f_best.split('/')[-1].split('_')[2:])

    filehandle.writelines("\n Train Accuracy is %s " % round(float(best_score['Train Accuracy']),2))
    filehandle.writelines("\n Validation Accuracy is %s " % round(float(best_score['Validation Accuracy']),2))
    filehandle.writelines("\n Test Accuracy is %s " % round(float(test_acc),2))
    # print("Best set of hyper parameters are \n:")
    # for attr, value in sorted(p_best.items()):
    #         print("\t{} = {}".format(attr.upper(), value))
    print('\n\n',"With Performance of \n")
    for attr, value in sorted(best_score.items()):
            print("\t{} = {:.2f}".format(attr, float(value)))
            print("Test Accuracy = ", round(float(test_acc),2))