import torch.nn.functional as F
import torch
import numpy as np
import multiprocessing

def bpr(batch_size, user, pos_item, neg_item, regularization):
    pos_score = torch.sum(torch.mul(user, pos_item), dim=1)
    neg_score = torch.sum(torch.mul(user, neg_item), dim=1)
    loss_term = F.logsigmoid(pos_score-neg_score)
    bpr_loss = -torch.mean(loss_term)
    regularizer = 1./2*(user**2).sum() + 1./2*(pos_item**2).sum() + 1./2*(neg_item**2).sum()
    regularizer = regularizer // batch_size
    regularizer_loss = regularizer * regularization
    return bpr_loss + regularizer_loss


def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


train_generator = None
test_dataset = None
def test(model, train_module, test_dataloader, pool):
    model.eval()
    correct = []
    global train_generator
    global test_dataset
    train_generator = train_module

    test_dataset = test_dataloader.dataset
    with torch.no_grad():
        user_embedding, item_embedding = model(train_module.H)
    n_user_batch = len(test_dataset.test_users) // train_module.bach_size + 1
    total_hit_rate = []
    for index, batch_test in enumerate(test_dataloader):
        user_batch = batch_test[index]
        user_embed = user_embedding[user_batch]
        item_embed = item_embedding[torch.range(0, train_module.n_item)]
        rate_batch = torch.matmul(user_embed, torch.transpose(item_embed, 0, 1))

        # Train에 나왔던 item은 뽑히지 않게
        make_pos_minus(user_batch, rate_batch, train_module.train_items)

        topk_item = torch.topk(rate_batch, k=train_module.topk, dim=1).indices
        topk_item = topk_item.detach().cpu().numpy()
        
        user_batch_rating = zip(topk_item, user_batch)
        result = pool.map(test_one_user, user_batch_rating)
        
        total_hit_rate += result
    return np.sum(total_hit_rate)/len(total_hit_rate)



def make_pos_minus(user_batch, rate_batch, pos_train_item):
    for i, user in enumerate(user_batch):
        pos_item = pos_train_item[user]
        rate_batch[i, pos_item] = -10000 

def test_one_user(x):
    topk_item = x[0]
    user = x[1]
    training_items = train_generator.train_items[user]
    pos_test_items = test_dataset.test_items[user]
    hit_list = []
    for pos in pos_test_items:
        if pos in topk_item:
            hit_list.append(1)
        else:
            hit_list.append(0)
    return hit_list





   
