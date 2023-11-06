import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, average_precision_score


def revise_trainMask(train_mask, gt, candidate_size=2):
    random.seed(121)
    for i in range(train_mask.shape[0]):
        # 從"未評論過的項目" 中，排除 gt 的 idx，之後取部分加到 mask 中
        test_idx = torch.where( gt[i]==1 )[0].tolist()
        not_interact_idx = torch.where( train_mask[i]==0 )[0].tolist()
        candidate_idx = list(set(not_interact_idx) - set(test_idx))
        # 要選幾個
        select_num = min( int(torch.sum(train_mask[i])*candidate_size ), len(candidate_idx))  # 目前評論數最多 158 個
        mask_num = len(candidate_idx) - select_num
        mask_idx = random.sample(candidate_idx, mask_num)
        train_mask[i][mask_idx] = 1
    
    return train_mask

def evaluation(all_user_logits, add_neg, pred_all, candidate_size, top_k=10):
    data_path = 'data/preprocessed_data.pkl' if add_neg==False else 'data/Exp_data/preprocessed_data.pkl'
    data = pd.read_pickle(data_path)

    trainMaskVector, gt_items = data['trainMaskVector'].clone(), list(data['test_data'].values())
    user_num, item_num = data['userCount'], data['itemCount']
    id_item_map = { idx:aid for idx,aid in enumerate(list(data['item_id_map'].keys())) }
    # 處理 ground truth (gt)
    gt = [ [ 1 if (id_item_map[j] in gt_items[i]) else 0 for j in range(item_num) ] for i in range(user_num) ]
    gt = torch.tensor(gt)

    if pred_all == 0:
        trainMaskVector = revise_trainMask(trainMaskVector, gt, candidate_size)
    trainMaskVector = torch.where(trainMaskVector>0, -100, 0)

    # 處理 prediction
    all_user_logits += trainMaskVector
    n_score = ndcg_score(gt, all_user_logits, k=top_k, ignore_ties=True)
    # m_score = average_precision_score(gt, all_user_logits, average='samples')
    # sort
    sorted_idx = torch.argsort(all_user_logits, dim=1, descending=True)
    top_idx = sorted_idx[:, :top_k]
    all_pred = [ [ 1 if (j in top_idx[i]) else 0 for j in range(item_num) ] for i in range(user_num) ]

    # hit ratio & MAP
    hit, m_score = 0, 0
    top_items = top_idx.clone().apply_(lambda x: id_item_map[x])
    for i, t_items in enumerate(top_items):
        if len( set(np.array(t_items)) & set(gt_items[i]) ) > 0 : hit += 1  # ???
        ap = sum([ 1/(j+1) if ele in gt_items[i] else 0 for j, ele in enumerate(t_items) ]) / len(gt_items[i])
        m_score += ap
    m_score /= user_num

    # evaluate by sklearn
    p_score = precision_score(gt, all_pred, average='samples', zero_division=0)
    r_score = recall_score(gt, all_pred, average='samples', zero_division=0)
    f_score = f1_score(gt, all_pred, average='samples', zero_division=0)
    eva_metrics = [ p_score, r_score, f_score, n_score, m_score, hit/user_num ]

    print(f'Top {top_k:<2} | precision:{p_score:.4f}, recall:{r_score:.4f}, f1:{f_score:.4f}, ndcg:{n_score:.4f}, map:{m_score:.4f}, hit:{hit/user_num:.4f}')

    return top_items, gt_items, eva_metrics