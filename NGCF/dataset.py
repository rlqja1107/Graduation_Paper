from utility import *
import pandas as pd
import pickle
import scipy.sparse as sp
import numpy as np
import random
from torch.utils.data import Dataset
import torch

class DataTest(Dataset):

    def __init__(self, config):
        self.n_test = 0
        self.test_items = {}
        self.path = config['path']
        
    def __len__(self):
        return len(self.test_items.keys())

    def __getitem__(self, index):
        return self.test_users[index]

    def load_test(self, n_item):
        with open(self.path+'/test.txt') as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                self.n_test += len(items[1:])
                self.test_items[items[0]] = items[1:]
        self.test_users = torch.LongTensor(list(self.test_items.keys()))


class DataTrain(Dataset):
    def __init__(self, config):
        self.path = config['path']
        self.batch_size = config['batch_size']
        self.topk = config['topk']
        self.n_user = 0
        self.n_item = 0
        self.n_train = 0
        self.train_items = {}
        self.test_items = {}
        self.R = sp.dok_matrix((10000000, 10000000), dtype=np.float32)
        self.n_neg_item = config['neg_item']
        self.pos_item = []
        self.neg_item = []
        self.hgnr = config['hgnr']

    def __len__(self):
        return len(self.exist_users)

    def __getitem__(self, index):
        return self.exist_users[index], self.pos_item[index], self.neg_item[index]

    def load_data(self):
        self.exist_users = []
        
        with open(self.path+'/train.txt') as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                self.R[uid, items] = 1.0
                self.train_items[uid] = items
                self.exist_users.append(uid)
                self.n_item = max(self.n_item, max(items))
                self.n_user = max(self.n_user, uid)
                self.n_train += len(items)

        with open(self.path+'/test.txt') as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                self.n_item = max(self.n_item, max(items[1:]))
        self.n_user += 1
        self.n_item += 1
        if self.hgnr:
            self.S = sp.dok_matrix((self.n_user, self.n_user), dtype=np.float32)
            with open(self.path+'/social_relations.txt') as f:
                for l in f.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    friends = [int(i) for i in l.split(' ')]
                    uid, friend_ids = friends[0], friends[1:]
                    for i in friend_ids:
                        self.S[uid, i] = 1.0

            self.C = sp.dok_matrix((self.n_item, self.n_item), dtype=np.float32)
            with open(self.path+'/review_idx_epi_g.txt') as f:
                for l in f.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    items = [int(i) for i in l.split(',')]
                    iid, item_ids = items[0], items[1:]
                    for i in item_ids:
                        self.C[iid, i] = 1.0
                
        
        self.R.resize((self.n_user, self.n_item))
        self.H = sp.dok_matrix((self.n_user+self.n_item, self.n_user+self.n_item), dtype = np.float32)
        self.normalize_H()
               
    def normalize_H(self):
        self.H = self.H.tolil()
        self.R = self.R.tolil()
        self.H[:self.n_user, self.n_user:] = self.R
        self.H[self.n_user:, :self.n_user] = self.R.T
        # item과 user matrix를 넣으면 됨.
        if self.hgnr:
            self.H[:self.n_user, :self.n_user] = self.S.tolil()
            self.H[self.n_user:, self.n_user:] = self.C.tolil()
        self.H = self.H.todok()

        # Normalize 
        rowsum = np.array(self.H.sum(1))
        D_inv = np.power(rowsum, -1/2).flatten()
        D_inv[np.isinf(D_inv)] = 0.0
        D_ = sp.diags(D_inv)
        self.L_c = (D_.dot(self.H)).dot(D_)
        self.L_c = sparse_mx_to_torch_sparse_tensor(self.L_c).cuda()


    def make_batch_sampling(self):
        pos_item, neg_item = [], []
        for user in self.exist_users:
            pos_item += [random.choice(self.train_items[user])]
            neg_item += self.neg_sampling(user)
        self.pos_item = np.asarray(pos_item)
        self.neg_item = np.asarray(neg_item)


    def neg_sampling(self, user):
        neg = []
        while True:
            if len(neg) >= self.n_neg_item:
                break
            rand = np.random.randint(0, self.n_item, 1)[0]
            if rand not in self.train_items[user] and rand not in neg:
                neg.append(rand)
        return neg
