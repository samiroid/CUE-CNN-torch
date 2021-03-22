import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, em_num_u, em_num, em_dim_u, em_dim, class_num, ker_num, ker_siz, conv, dropout, words_pt, users_pt, words_vec, users_vec):
        super(CNN_Text, self).__init__()
        # self.args = args
        self.u_emb = True #accoutns for user embeddings
        V_u = em_num_u
        V = em_num
        D_u = 400
        # D_u = em_dim_u
        D = 400
        # D = em_dim
        C = class_num
        Ci = 1
        Co = ker_num
        Ks = ker_siz
        M = conv
        Dr = dropout
        # D_u = args.embed_dim_users
        
        self.embed = nn.Embedding(V, D)
        # self.embed_u = nn.Embedding(V_u, D_u)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(Dr)
        # self.fc1 = nn.Linear(len(Ks) * Co, M)
        # self.fc1 = nn.Linear(len(Ks) * Co, C)
        # self.fc2 = nn.Linear(M + D_u, C)

        if not self.u_emb:
            self.fc1 = nn.Linear(len(Ks) * Co, C)
        else:
            self.embed_u = nn.Embedding(V_u, D_u)
            self.fc1 = nn.Linear(len(Ks) * Co, M)
            self.fc2 = nn.Linear(M + D_u, C)

        if words_pt:
            self.embed.weight.data.copy_(words_vec)
            # print(words_vec.size())
            self.embed.weight.requires_grad = False
        if users_pt:
            self.embed_u.weight.data.copy_(users_vec)
            self.embed_u.weight.requires_grad = False
            # print("not trained")
            # D = 400

    def forward(self, x, u):
        x = self.embed(x)  # (N, W, D)
        # u = self.embed_u(u)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, M) c
        if self.u_emb:
            # print("users accounted")
            u = self.embed_u(u)
            x = torch.cat((x, u.squeeze(1)), dim = 1) #(M+D_u)
            x = self.fc2(x)  # (M+D_u, C)
        return x