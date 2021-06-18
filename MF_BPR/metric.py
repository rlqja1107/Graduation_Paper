import numpy as np
from sklearn.metrics import roc_auc_score

def auc_score(model, bpr_ratings):
    """
    computes area under the ROC curve.
    """
    auc = 0.0
    n_users, n_items = bpr_ratings.shape
    for user, row in enumerate(bpr_ratings):
        y_pred = model.predict_user(user)
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)

    auc /= n_users

    return auc

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item in gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem, k):
    dcg = 0.0

    if len(gtItem) >= k:
      idcg = sum([1.0/np.log(i+1) for i in range(1, k+1)])
    else:
      idcg = sum([1.0/np.log(i+1) for i in range(1, len(gtItem)+1)])

    for i, r in enumerate(ranklist):
        if r in gtItem:
            dcg += 1.0/np.log(i+2)

    return dcg/idcg
