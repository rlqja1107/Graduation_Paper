from timeit import default_timer as timer
import torch.nn.functional as F
import numpy as np
import torch

def bpr(batch_size, user, pos_item, neg_item, regularization):
    pos_score = torch.sum(torch.mul(user, pos_item), dim=1)
    neg_score = torch.sum(torch.mul(user, neg_item), dim=1)
    loss_term = F.logsigmoid(pos_score-neg_score)
    bpr_loss = -torch.mean(loss_term)
    regularizer = 1./2*(user**2).sum() + 1./2*(pos_item**2).sum() + 1./2*(neg_item**2).sum()
    regularizer = regularizer / batch_size
    regularizer_loss = regularizer * regularization
    return bpr_loss, regularizer_loss


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def test(model, train_module, test_dataloader):
    model.eval()
    test_dataset = test_dataloader.dataset
    topk_result = dict()
    with torch.no_grad():
        user_embedding, item_embedding = model(train_module.L_c)
    total_hit_rate = []
    total_ndcg = []
    for index, batch_test in enumerate(test_dataloader):
        user_batch = batch_test.cuda()
        user_embed = user_embedding[user_batch]
        item_embed = item_embedding[torch.arange(0, train_module.n_item, dtype = torch.long).cuda()]
        rate_batch = torch.matmul(user_embed, torch.transpose(item_embed, 0, 1))
        rate_batch = rate_batch.detach()
        # Train에 나왔던 item은 뽑히지 않게
        make_pos_minus(user_batch, rate_batch, train_module.train_items)

        topk_item = torch.topk(rate_batch, k=train_module.topk, dim=1).indices
        topk_item = topk_item.cpu().numpy()
        
        for i, user in enumerate(user_batch):
            hit, ndcg = test_one_user(user.item(), topk_item[i], test_dataset)
            total_hit_rate.append(hit)
            total_ndcg.append(ndcg)
            
        temp_dict = dict(zip(user_batch.detach().cpu().numpy(), topk_item))
        topk_result.update(temp_dict)

    return np.sum(total_hit_rate)/test_dataset.test_user, np.sum(total_ndcg)/test_dataset.test_user, topk_result



def make_pos_minus(user_batch, rate_batch, pos_train_item):
    for i, user in enumerate(user_batch):
        user = user.item()
        pos_item = pos_train_item[user]
        rate_batch[i, pos_item] = -10000.0 

def test_one_user(user, topk_item, test_dataset):
    # hit rate
    pos_test_items = test_dataset.test_items[user]
    hit_list = []
    for pos in topk_item:
        if pos in pos_test_items:
            hit_list.append(1)
        else:
            hit_list.append(0)
    hit_rate = 1 if np.sum(hit_list) > 0 else 0
    # NDCG
    reverse_list = np.ones(len(hit_list))
    IDCG = np.sum(reverse_list / np.log2(np.arange(2, len(reverse_list)+2)))
    DCG = np.sum(hit_list / np.log2(np.arange(2, len(hit_list)+2)))
    NDCG = 0.0 if DCG == 0.0 else DCG/IDCG
        
    return hit_rate, NDCG