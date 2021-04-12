import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import pandas as pd
import random

from itertools import product
import numpy as np

random.seed(42)

parser = argparse.ArgumentParser(description='CNN hyper parameter tuning')
parser.add_argument('-pretrained-embed-words', type=bool, default=True, help='Use pre-trained embedding for words')
parser.add_argument('-pretrained-embed-users', type=bool, default=False, help='Use pre-trained embedding for users')
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


p = {
      'lr': [0.001, 0.01],
      'dropout':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
      'max_norm':[0.003, 0.001, 0.0006, 0.0003, 0.0001],
      'kernel_num':np.linspace(40,100,61),
      'kernel_size':['2,3,4','5,6,7','8,9,10','11,12,13','14,15,16','2,4,6','3,5,7'],
      'conv_layer':np.linspace(50,200, 31)
      }

best_score = {'Train Accuracy': 0, 'Validation Accuracy': 0}
# function product takes cartesian product
# list of dicts
p_list = list((dict(zip(p.keys(), values)) for values in product(*p.values())))
random.shuffle(p_list)


# load MR dataset
def mr(text_field, label_field, user_field, **kargs):
    train_data, dev_data, test_data = mydatasets.MR.splits(text_field, label_field, user_field, args = args)
    if args.pretrained_embed_words:
        text_field.build_vocab(train_data, dev_data, test_data, vectors = args.custom_embed)
        print(args.custom_embed)
    else:
        text_field.build_vocab(train_data, dev_data, test_data)
    if args.pretrained_embed_users:
        user_field.build_vocab(train_data, dev_data, test_data, vectors = args.custom_embed_u)
    else:
        user_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    # print(train_data)
    # split valid and train (10%)

    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train_data, dev_data, test_data), 
                                batch_sizes=(batch_size, len(dev_data), len(test_data)),
                                **kargs)
    return train_iter, dev_iter, test_iter


with open('results_4.txt', 'w') as filehandle:
    filehandle.writelines("CNN with not pretrained user embeddings and pretrained word embeddings\n")
    # filehandle.writelines("%s " % attr for attr,_ in best_score.items())
    filehandle.writelines("Tr_acc ")
    filehandle.writelines("Val_acc ")
    # filehandle.writelines(" ")
    filehandle.writelines("%s " % attr for attr,_ in p_list[0].items())    

    for i_dict in p_list[:100]:
        # learning
        lr = float(i_dict['lr'])
        epochs = 10
        log_interval = 1
        test_interval = 20
        save_interval = 500
        save_dir = 'snapshot'
        early_stop = 100
        save_best = False

        # model
        dropout = float(i_dict['dropout'])
        max_norm = float(i_dict['max_norm'])
        embed_dim = 128
        embed_dim_users = 128
        kernel_num = int(i_dict['kernel_num'])
        kernel_sizes = i_dict['kernel_size']
        # pretrained_embed_words = True
        # pretrained_embed_users = True
        conv_layer = int(i_dict['conv_layer'])
        batch_size = 128
        # device
        # data 


        # load data
        # print("\nLoading data...")
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        user_field = data.Field()
        train_iter, dev_iter, test_iter = mr(text_field, label_field, user_field, device='cpu', repeat=False)
        
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
            for attr, value in sorted(i_dict.items()):
                print("\t{} = {}".format(attr.upper(), value))

            print('\n ############################################################ \n')
            if dev_acc > best_score['Validation Accuracy']:
                test_acc = train.eval(test_iter, cnn, args) 
                best_score['Train Accuracy'] = tr_acc
                best_score['Validation Accuracy'] = dev_acc
                p_best = i_dict.copy()
                # best_sct.copy()

        filehandle.writelines("\n")
        filehandle.writelines("%s " % round(float(tr_acc),2))
        filehandle.writelines("%s " % round(float(dev_acc),2))
        filehandle.writelines("%s " % val for _,val in i_dict.items())
        # filehandle.writelines("%s " % val for _,val in .items())
        
    filehandle.writelines('\n\n' + '-' * 89)
    filehandle.writelines("\n Best set of hyper parameters\n")
    filehandle.writelines(" %s" % attr for attr,_ in p_best.items())
    filehandle.writelines("\n")
    filehandle.writelines(" %s" % val for _,val in p_best.items())

    filehandle.writelines("\n Train Accuracy is %s " % round(float(best_score['Train Accuracy']),2))
    filehandle.writelines("\n Validation Accuracy is %s " % round(float(best_score['Validation Accuracy']),2))
    filehandle.writelines("\n Test Accuracy is %s " % round(float(test_acc),2))
    print("Best set of hyper parameters are \n:")
    for attr, value in sorted(p_best.items()):
            print("\t{} = {}".format(attr.upper(), value))
    print('\n\n',"With Performance of \n")
    for attr, value in sorted(best_score.items()):
            print("\t{} = {:.2f}".format(attr, float(value)))
            print("Test Accuracy = ", round(float(test_acc),2))