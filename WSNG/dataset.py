from utility import *
import pandas as pd
import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset
import torch
from timeit import default_timer as timer

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
        self.test_n_user = 0
        with open(self.path+'/test_3.txt') as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                self.n_test += len(items[1:])
                self.test_items[items[0]] = items[1:]
                self.test_n_user+=1
        self.test_users = torch.LongTensor(list(self.test_items.keys()))


class DataTrain(Dataset):
    def __init__(self, config, batch_size):
        self.path = config['path']
        self.batch_size = batch_size
        self.topk = config['topk']
        self.n_user = 0
        self.n_item = 0
        self.n_train = 0
        self.train_items = {}
        self.test_items = {}
        self.n_neg_item = config['neg_item']
        self.pos_item = []
        self.neg_item = []
        self.hgnr = config['hgnr']
        self.sample_users = []

    def __len__(self):
        return self.n_train

    def __getitem__(self, index):
        return self.sample_users[index], self.pos_item[index], self.neg_item[index]

    def load_data(self):
        data = {}
        with open(self.path+'/epinion_time_3.txt', 'r') as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                data[(int(l[0]), int(l[1]))] = int(l[2])
        exist_users = []
        train_items = {}
        n_item = 0 
        n_user = 0
        n_train = 0
        train_list = []
        with open(self.path+'/train_3.txt', 'r') as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                exist_users.append(uid)
                train_items[uid] = items
                n_item = max(n_item, max(items))
                n_user = max(n_user, uid)
                n_train += len(items)
                for i in items:
                    train_list.append([uid, i, data[(uid, i)]])
        self.train_items = train_items
        with open(self.path+'/test_3.txt') as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                n_item = max(n_item, max(items[1:]))
        n_user += 1
        n_item += 1
        self.n_user = n_user
        self.n_item = n_item
        self.total_length = n_user + n_item

        train = pd.DataFrame(train_list, columns = ['user', 'item', 'time'])
        self.train_sort = train.sort_values(by = ['time'], axis= 0)
        
        self.share = n_train // self.batch_size
        self.rest = n_train % self.batch_size
        self.construct_batch()
    
    def construct_batch(self):
        batch_list = []
        adj_list = []
        print("Make Adjacency Matrix")
        start =timer()
        for i in range(self.share):
            f_time = self.train_sort.iloc[i*self.batch_size][2]
            l_time = self.train_sort.iloc[(i+1)*self.batch_size-1][2]
            # time_df = self.train_sort[(self.train_sort['time']<=l_time)&(self.train_sort['time']>=f_time)]
            time_df = self.train_sort[self.train_sort['time']<=l_time]
            r1 = np.asarray(time_df['user'])
            c1 = np.asarray(time_df['item']) + self.n_user
            r2 = np.asarray(time_df['item']) + self.n_user
            c2 = np.asarray(time_df['user'])
            r = np.hstack([r1,r2])
            c = np.hstack([c1,c2])
            adj = sp.csr_matrix((np.ones(len(r)),(r,c)),shape = [self.total_length, self.total_length])
            adj = self.normalize_H(adj)
            adj_list.append(adj)
            batch_list.append((self.train_sort.iloc[i*self.batch_size:(i+1)*self.batch_size]).values)

        batch_list.append((self.train_sort.iloc[self.share*self.batch_size:]).values)
        # f_time = self.train_sort.iloc[i*self.batch_size][2]
        # time_df = self.train_sort[self.train_sort['time']>=f_time]
        f_time = self.train_sort.iloc[len(self.train_sort)-1][2]
        time_df = self.train_sort[self.train_sort['time']<=f_time]
        r1 = np.asarray(time_df['user'])
        c1 = np.asarray(time_df['item']) + self.n_user
        r2 = np.asarray(time_df['item']) + self.n_user
        c2 = np.asarray(time_df['user'])
        r = np.hstack([r1,r2])
        c = np.hstack([c1,c2])
        adj = sp.csr_matrix((np.ones(len(r)),(r,c)),shape = [self.total_length, self.total_length])
        adj = self.normalize_H(adj)
        adj_list.append(adj)
        self.batch_list = batch_list
        self.adj_list = adj_list
        print("Finish adj mtx : {:4f}".format(timer()-start))
        
    def normalize_H(self, H):
        if self.hgnr:
            self.H[:self.n_user, :self.n_user] = self.S.tolil()
            self.H[self.n_user:, self.n_user:] = self.C.tolil()
        self.H = H.todok()

        # Normalize 
        rowsum = np.array(self.H.sum(1))
        D_inv = np.power(rowsum, -1/2).flatten()
        D_inv[np.isinf(D_inv)] = 0.0
        D_ = sp.diags(D_inv)
        L_c = (D_.dot(self.H)).dot(D_)
        return sparse_mx_to_torch_sparse_tensor(L_c).cuda()




    def neg_sampling(self,user, dic):
        neg = []
        while True:
            if len(neg) >= 1:
                break
            rand = np.random.randint(0, self.n_item, 1)[0]
            if (user,rand) not in dic and rand not in neg:
                neg.append(rand)
        return neg
