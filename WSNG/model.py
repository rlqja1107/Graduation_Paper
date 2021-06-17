import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, n_embedding):
        super(LSTM, self).__init__()
        self.n_embedding = n_embedding
        self.gru_1_layer = torch.nn.GRU(input_size = 64, hidden_size = 64, num_layers = 1)
        self.gru_2_layer = torch.nn.GRU(input_size = 64, hidden_size = 64, num_layers = 1)
        self.gru_3_layer = torch.nn.GRU(input_size = 64, hidden_size = 64, num_layers = 1)
        self.front_prev_hidden = []
        self.back_prev_hidden = []
        for i in range(3):
            self.front_prev_hidden.append(torch.rand(1,64,64).cuda())
            self.back_prev_hidden.append(torch.rand(1,64,64).cuda())
            nn.init.xavier_normal_(self.front_prev_hidden[i])
            nn.init.xavier_normal_(self.back_prev_hidden[i])


    def forward(self, num, front = True):
        if num == 0:
            _, h = self.gru_1_layer(self.front_prev_hidden[num], self.front_prev_hidden[num]) if front else self.gru_1_layer(self.back_prev_hidden[num], self.back_prev_hidden[num])
            return h
        elif num == 1:
            _, h =  self.gru_2_layer(self.front_prev_hidden[num], self.front_prev_hidden[num]) if front else self.gru_2_layer(self.back_prev_hidden[num], self.back_prev_hidden[num])
            return h
        else:
            _, h =  self.gru_3_layer(self.front_prev_hidden[num], self.front_prev_hidden[num]) if front else self.gru_3_layer(self.back_prev_hidden[num], self.back_prev_hidden[num])
            return h



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
            # if i == 2:
            #     final_node_embed = normalize_embed
            all_embedding += [normalize_embed]
        
        all_embedding = torch.cat(all_embedding, dim=1)
        # self.user_embedding.weight.data, self.item_embedding.weight.data = torch.split(final_node_embed, [self.n_user, self.n_item], dim = 0)
        user_embedding, item_embedding = torch.split(all_embedding, [self.n_user, self.n_item], dim=0)
        return user_embedding, item_embedding
            
