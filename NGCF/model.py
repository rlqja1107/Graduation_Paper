import torch.nn as nn
import torch
import numpy as np

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, n_embedding, dropout):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_embedding = n_embedding
        self.user_embedding = nn.Embedding(self.n_user, self.n_embedding)
        self.item_embedding = nn.Embedding(self.n_item, self.n_embedding)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.Front_Linear_List = nn.ModuleList()
        self.Back_Linear_List = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for _ in range(3):
            self.Front_Linear_List.append(nn.Linear(n_embedding, n_embedding))
            self.Back_Linear_List.append(nn.Linear(n_embedding, n_embedding))
            self.dropout_list.append(nn.Dropout(p=dropout))

    def forward(self, H):
        E_l_embedding = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim = 0)
        all_embedding = [E_l_embedding]
        H_I = H
        for i in range(3):
            Front = torch.sparse.mm(H_I, E_l_embedding)
            Front_cal = nn.functional.leaky_relu(self.Front_Linear_List[i](Front)+ self.Front_Linear_List[i](E_l_embedding))
            Back = torch.mul(Front, E_l_embedding)
            Back = nn.functional.leaky_relu(self.Back_Linear_List[i](Back))
            E_l_embedding = Front_cal + Back
            E_l_embedding = self.dropout_list[i](E_l_embedding)
            normalize_embed = nn.functional.normalize(E_l_embedding, p=2, dim=1)
            all_embedding += [normalize_embed]
        
        all_embedding = torch.cat(all_embedding, dim=1)
        user_embedding, item_embedding = torch.split(all_embedding, [self.n_user, self.n_item], dim=0)
        return user_embedding, item_embedding
            
