# load Sarcasm dataset
import mydatasets
import torch
import torchtext.legacy.data as data
from torchtext import vocab

batch_size = 128
def dataloader(text_field, label_field, user_field, args, wdir=None, u2vdir=None, **kargs):
    train_data, dev_data, test_data = mydatasets.MR.splits(text_field, label_field, user_field, args = args)
    if args.pretrained_embed_words:
        custom_embed = vocab.Vectors(name = wdir, max_vectors = 100000)
        text_field.build_vocab(train_data, dev_data, test_data, vectors = custom_embed)
        # print(args.custom_embed)
    else:
        text_field.build_vocab(train_data, dev_data, test_data)
    if args.pretrained_embed_users:
        custom_embed_u = vocab.Vectors(name = u2vdir, max_vectors = 8000)
        user_field.build_vocab(train_data, dev_data, test_data, vectors = custom_embed_u)
    else:
        user_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    # split valid and train (10%)

    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train_data, dev_data, test_data), 
                                batch_sizes=(batch_size, len(dev_data), len(test_data)),
                                **kargs)
    return train_iter, dev_iter, test_iter
