import pandas as pd
import pickle
import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset

class EpinionTest(Dataset):
    def __init__(self, config):
        self.n_test = 0
        self.test_items = {}
        self.path = config['path']
        
    def __len__(self):
        return len(self.test_items.keys())

    def __getitem__(self, index):
        return self.test_users[index]

    def load_test(self):
        with open(self.path+'/test.txt') as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                self.n_test += len(items[1:])
                self.test_items[items[0]] = items[1:]
        self.test_users = list(self.test_items.keys())

        

class EpinionData(Dataset):
    def __init__(self, config):
        self.path = config['path']
        self.batch_size = config['batch_size']
        self.topk = config['topk']
        self.n_user = 0
        self.n_item = 0
        self.n_train = 0
        self.n_test = 0
        self.train_items = {}
        self.R = sp.dok_matrix((1000000, 1000000), dtype=np.float32)
        self.neg_item = config['neg_item']
        self.pos_item = []
        self.neg_item = []

    def __len__(self):
        return self.n_train

    def __getitem__(self, index):
        return self.exist_users[index], self.pos_item[index], self.neg_item[index]

    def load_data(self):
        self.item_list = pd.read_table(self.path+'/item_list.txt', sep = '\t', header=0)
        self.user_list = pd.read_table(self.path+'/user_list.txt', sep = '\t', header=0)
        self.exist_users = []
        
        with open(self.path+'/train.txt') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    for item in items:
                        self.R[uid, item] = 1.0
                    self.train_items[uid] = items
                    self.exist_users.append(uid)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.n_train += len(items)
        
        self.n_user += 1
        self.n_item += 1
        self.R = self.R.reshape(shape=(self.n_user, self.n_item))

      
        self.Item_By_Item = sp.dok_matrix((self.n_item, self.n_item), dtype = np.float32)
        self.User_By_User = sp.dok_matrix((self.n_user, self.n_user), dtype = np.float32)
        self.H = sp.dok_matrix((self.n_user+self.n_item, self.n_user+self.n_item), dtype = np.float32)

               
    def normalize_H(self):
        self.H = self.H.tolil()
        self.R = self.R.tolil()
        self.H[:self.n_user, self.n_user:] = self.R
        self.H[self.n_user:, :self.n_user] = self.R.T
        # item과 user matrix를 넣으면 됨.
        # self.H[:self.n_user, :self.n_user] = User Matrix
        # self.H[self.n_user:, self.n_user:] = Item Matrix
        self.H = self.H.todok()

        # Normalize 
        H_hat = self.H + sp.eye(self.H.shape[0])
        rowsum = np.array(self.H.sum(1))
        D_inv = np.power(rowsum, -1).flatten()
        D_inv[np.isinf(D_inv)] = 0.0
        D_ = sp.diags(D_inv)
        self.L_c = (D_.dot(H_hat)).dot(D_)


    def make_batch_sampling(self):
        pos_item, neg_item = [], []
        for user in self.exist_users:
            pos_item += np.random.choice(self.train_items[user], 1)
            neg_item.append(self.neg_sampling(user))
        self.pos_item = np.asarray(pos_item)
        self.neg_item = np.asarray(neg_item)


    def neg_sampling(self, user):
        neg = []
        while True:
            if len(neg) >= self.neg_item:
                break
            rand = np.random.randint(0, self.n_item, 1)
            if rand not in self.train_items[user] and rand not in neg:
                neg.append(rand)
