import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

class Neighbor_based_attention(nn.Module):
    def __init__(self, item_count, num_heads):
        super(Neighbor_based_attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=item_count, num_heads=num_heads, batch_first=True)
        
    def forward(self, user_batch, neighbor_batch):
        # batch.shape => (batch, item_count)
        user_batch = user_batch.unsqueeze(1).to(torch.float32) #(batch, seq_len, item_count)
        neighbor_batch = neighbor_batch.unsqueeze(1).to(torch.float32) #(batch, seq_len, item_count)
        attn_output, attn_output_weights = self.multihead_attn(user_batch, neighbor_batch, neighbor_batch) # (batch, seq_len, item_count)
        attn_output = attn_output.squeeze(1) # (batch, item_count)
        
        return attn_output

class Discriminator(nn.Module):
    def __init__(self, itemCount):
        super(Discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        D_input = data
        result = self.dis(D_input)
        return result
    
class Generator_nbr(nn.Module):
    def __init__(self, itemCount, num_heads):
        self.itemCount = itemCount
        self.num_heads = num_heads
        super(Generator_nbr,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(self.itemCount, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Sigmoid()
        )
        self.nbr_attn = Neighbor_based_attention(item_count=self.itemCount, num_heads=self.num_heads)
        
    def forward(self, noise, user_pref_vector, user_neighbors_ratings_batch):
        G_input = user_pref_vector
        result = self.gen(G_input)
        result = self.nbr_attn(user_batch=result, neighbor_batch=user_neighbors_ratings_batch)
        result = self.gen(result)
        
        return result

def result_plt(result_precision, plot_name='test'):
    plt.figure()
    plt.xlabel('epoch')
    plt.plot(result_precision[:,0], result_precision[:,1], "r")
    plt.title(plot_name)
    # plt.savefig(f'model/result/200ep/{plot_name}.png')
    plt.show()

def select_negative_items(realData, itemCount, user_neighbors_ratings_batch, num_lm, num_nbrm, num_zr):

    data = realData.detach().cpu()
    nbr_data = np.array(user_neighbors_ratings_batch.detach().cpu())

    _, sorted_logits_idx = torch.sort(data, dim=1, descending=True)
    min_logits_idx = np.array(sorted_logits_idx[:, -num_lm:])
    data = np.array(data)
    n_items_lm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    n_items_nbrm = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        nbr_p_items = np.where(nbr_data[i] >= 4)[0]
        all_item_index = random.sample(range(data.shape[1]), itemCount)
        all_nbr_item_index = random.sample(range(nbr_data.shape[1]), itemCount)

        for j in range(p_items.shape[0]):
            all_item_index.remove(list(p_items)[j])
        for j in range(nbr_p_items.shape[0]):
            all_nbr_item_index.remove(list(nbr_p_items)[j])
        
        random.shuffle(all_item_index)
        random.shuffle(all_nbr_item_index)
        n_item_index_zr = all_item_index[0 : num_zr]
        n_item_index_nbrm = all_nbr_item_index[0 : num_nbrm]
        
        n_items_lm[i][min_logits_idx[i]] = 1
        n_items_zr[i][n_item_index_zr] = 1
        n_items_nbrm[i][n_item_index_nbrm] = 1
        
    
    return n_items_lm, n_items_nbrm, n_items_zr

def select_negative_items_CF(realData, itemCount, num_pm, num_zr):

    data = realData.detach().cpu()
    data = np.array(data)
    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        all_item_index = random.sample(range(data.shape[1]), itemCount)
        for j in range(p_items.shape[0]):
            all_item_index.remove(list(p_items)[j])
        random.shuffle(all_item_index)

        n_item_index_pm = all_item_index[0 : num_pm]
        n_item_index_zr = all_item_index[num_pm : (num_pm+num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
        
    return n_items_pm, n_items_zr

def computeTopK(groundTruth, result, top_k, item_id_map):
    # 因為使用者互動的商品最少只有 30 個，test 取0.2的話就只有6個 -> 算 Top 10 沒意義
    
    batch_precision, batch_recall = 0, 0
    batch_ndcg, batch_ap = 0, 0

    # groundTruth[i] 存 user i 在 test 中的 AppID，result 存對每個商品的機率
    for i, res in enumerate(result):
        gt = groundTruth[i]
        # 找出 AppID 所在的 index   
        gt = [item_id_map[g] for g in gt]
        res = res.tolist()

        # i 表示 app_id 的 index
        res = [ (res[i], i) for i in range(len(res)) ]
        res.sort(key=lambda x:x[0], reverse=True)

        hit, dcg, idcg, sum_precs = 0, 0, 0, 0
        idcgCount = len(gt)

        for j in range(top_k):
            if(res[j][1] in gt):
                hit += 1
                dcg += 1 / math.log2(j+2)
                sum_precs += hit / (j+1)

            if(idcgCount>0):
                idcg += 1/math.log2(j+2)
                idcgCount -= 1

        if len(gt)!=0 and top_k!=0:
            precision = hit / top_k
            recall = hit / len(gt)
            ndcg = dcg / idcg
            ap = sum_precs / len(gt)
        else:
            precision, recall = 0, 0
            ndcg, ap = 0, 0
        
        batch_precision += precision
        batch_recall += recall
        batch_ndcg += ndcg
        batch_ap += ap
    
    return batch_precision, batch_recall, batch_ndcg, batch_ap

def train_GAN(config, b_seed=None):
    data_path = 'data/preprocessed_data.pkl' if config['add_neg']==False else 'data/Exp_data/preprocessed_data.pkl'
    data = pd.read_pickle(data_path)
    
    best_ep = 0
    trainVector, trainMaskVector, test_data = data['trainVector'], data['trainMaskVector'], data['test_data']
    test_data = list(test_data.values())
    item_id_map, itemCount = data['item_id_map'], data['itemCount']
    
    # user 相關
    user_emb = pd.read_pickle('data/User/user_id_emb.pkl')
    all_user_id = list(user_emb.keys())
    all_user_logits = pd.read_pickle('data/User/all_user_logits.pkl')
    user_revEmb = pd.read_pickle('data/User/user_id_ReviewsEmb.pkl')
    user_itemEmb = pd.read_pickle('data/User/user_interacted_appEmb.pkl')
    user_nbr_revEmb = pd.read_pickle('data/User/user_id_nbrReviewsEmb.pkl')
    user_neighbors_ratings = pd.read_pickle('data/User/user_id_nbrRating.pkl')

    # config
    # ---------------------
    epochCount, batchSize = config['epochs'], config['batchSize']
    top_k = config['top_k']
    num_pm, num_zr = int(config['pm_rate']*itemCount), int(config['zr_rate']*itemCount)
    num_LM, num_nbrm = int(config['lm_rate']*itemCount), int(config['nbrm_rate']*itemCount)
    alpha = config['alpha']
    step_ratio = config['G/D_step_ratio']
    num_heads = config['num_heads']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ---------------------

    # 初始化評估參數
    top_ndcg = 0
    result_precision, result_recall = {}, {}
    result_ndcg, result_map = {}, {}
    for n in top_k:
        result_precision[n], result_recall[n] = np.zeros((1,2)), np.zeros((1,2))
        result_ndcg[n], result_map[n] = np.zeros((1,2)), np.zeros((1,2))
    g_loss_list, d_loss_list = np.zeros((1,2)), np.zeros((1,2))


    # Build the Neighbor Module
    # user side
    user_emb = torch.from_numpy(np.array(list(user_emb.values()))).to(device)  # [ 704, 512 ]
    user_revEmb = torch.from_numpy(np.array(list(user_revEmb.values()))).to(device)  # [ 704, 30, 512 ]
    user_itemEmb = torch.from_numpy(np.array(list(user_itemEmb.values()))).to(device)  # [ 704, 30, 512 ]
    user_nbr_revEmb = torch.from_numpy(np.array(list(user_nbr_revEmb.values()))).to(device)  # [ 704, 150, 512 ]
    user_neighbors_ratings = torch.from_numpy(np.array(list(user_neighbors_ratings.values()))).to(device)  # [ 704, 428 ]

    # Initialize Model
    torch.manual_seed(config['seed'])
    G_nbr = Generator_nbr(itemCount, num_heads=num_heads).to(device)
    torch.manual_seed(config['seed'])
    D = Discriminator(itemCount).to(device)

    # Loss criteria
    regularization = nn.MSELoss().to(device)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config['lr'])
    g_nbr_optimizer = torch.optim.Adam(G_nbr.parameters(), lr=config['lr'])
    
    # ---------------------
    # 開 train
    # ---------------------
    pbar = tqdm(range(epochCount), desc='Epoch ')
    for epoch in pbar:

        batchSize = config['batchSize']
        # 定義需要將資料切成幾等分的 batch，所以每個 batch 的資料個數可能會跟定義的 batch_size 不同
        # batches 是 list ，每個元素存 index | ex: [ 64, 65, ..., 127 ]。一個元素代表一個 user
        batches = np.array_split(np.arange(len(trainVector)), len(trainVector)/(batchSize-1))
        # 從 batches 中抽幾個 batch 來跑模型。最少抽 3 個 batch，最多一半
        # 當 batch_size 太大時，抽的數量超過 batch 數就會噴錯 !!!
        batches = random.sample( batches, max(3, len(batches)//2) )
        # random.shuffle(batches)
        
        step = 0
        change_step = int(len(batches)*step_ratio)
        batches_pbar = tqdm(batches, leave=False, desc='Batches')
        for batch in batches_pbar:
            leftIndex = batch[0]
            batchSize = len(batch)
            
            # Select a random batch of purchased vector
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize])).to(device)
            eu = Variable(copy.deepcopy(trainMaskVector[leftIndex:leftIndex + batchSize])).to(device)
            # 鄰居評分向量
            user_neighbors_ratings_batch = user_neighbors_ratings[leftIndex:leftIndex + batchSize].to(device)

            # ---------------------
            #  Train Generator
            # ---------------------
            if step < change_step:

                # Select a random batch of negative items for every user
                b_seed[0] += b_seed[1]
                random.seed(b_seed[0])
                if config['ab_test'] == False:
                    n_items_pm, n_items_nbrm, n_items_zr = select_negative_items(realData, itemCount, user_neighbors_ratings_batch, num_LM, num_nbrm, num_zr)
                    ku_zp = Variable(torch.tensor(n_items_pm + n_items_nbrm + n_items_zr))
                else:
                # ---------------------------------------------------------------------------
                    # Ablation w/o LM-NM
                    n_items_pm, n_items_zr = select_negative_items_CF(realData, itemCount, num_pm, num_zr)
                    ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
                # ---------------------------------------------------------------------------

                ku_zp[ku_zp >= 1] = 1
                ku_zp = ku_zp.to(device)
                realData_zp = Variable(torch.ones_like(realData)) * eu

                # Neighbor interacted module
                user_pref_vector = torch.tensor([]).to(device)
                for idx in range(leftIndex, leftIndex + batchSize):
                    user_id = all_user_id[idx]
                    # 取出單一使用者對所有遊戲的喜好機率
                    user_logits = all_user_logits[user_id].to(device)  # [1, 428]
                    user_pref_vector = torch.cat([user_pref_vector, user_logits], 0)  # [b, 428]

                # Generate a batch of new purchased vector
                fakeData_nbr = G_nbr(realData, user_pref_vector, user_neighbors_ratings_batch)
                fakeData = fakeData_nbr
                fakeData_ZP = fakeData * (eu + ku_zp)
                fakeData_result = D(fakeData_ZP)

                # Train the generator
                g_loss = np.mean(np.log(1.-fakeData_result.detach().cpu().numpy()+10e-5)) + alpha*regularization(fakeData_ZP, realData_zp)
                g_nbr_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                g_nbr_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            else:
                # print('D step')
                # Select a random batch of negative items for every user
                b_seed[0] += 1
                random.seed(b_seed[0])
                if config['ab_test'] == False:
                    n_items_lm, n_items_nbrm, _ = select_negative_items(realData, itemCount, user_neighbors_ratings_batch, num_LM, num_nbrm, num_zr)
                    ku = Variable(torch.tensor(n_items_lm + n_items_nbrm))
                else:
                # ---------------------------------------------------------------------------
                    # Ablation w/o LM-NM
                    n_items_pm, _ = select_negative_items_CF(realData, itemCount, num_pm, num_zr)
                    ku = Variable(torch.tensor(n_items_pm))
                # ---------------------------------------------------------------------------

                ku[ku >= 1] = 1
                ku = ku.to(device)
                realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku

                # Neighbor interacted module
                user_pref_vector = torch.tensor([]).to(device)
                for idx in range(leftIndex, leftIndex + batchSize):
                    user_id = all_user_id[idx]
                    # 取出單一使用者對所有遊戲的喜好機率
                    user_logits = all_user_logits[user_id].to(device)  # [1, 428]
                    user_pref_vector = torch.cat([user_pref_vector, user_logits], 0)  # [b, 428]

                # Generate a batch of new purchased vector
                fakeData_nbr = G_nbr(realData, user_pref_vector, user_neighbors_ratings_batch)
                fakeData = fakeData_nbr
                fakeData_ZP = fakeData * (eu + ku)

                # Train the discriminator
                fakeData_result = D(fakeData_ZP) 
                realData_result = D(realData)
                
                d_loss = -np.mean(np.log(realData_result.detach().cpu().numpy()+10e-5) + 
                                  np.log(1. - fakeData_result.detach().cpu().numpy()+10e-5)) + 0*regularization(fakeData_ZP, realData_zp)
                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

            step+=1


        # ---------------------
        #  Evaluate
        # ---------------------
        if( epoch%10 == 9 ):
            all_user_top = []
            n_user = len(test_data)
            batchSize = 128
            
            # Initialize evaluation metrics
            precisions, recalls = {}, {}
            ndcgs, aps = {}, {}
            for n in top_k:
                precisions[n], recalls[n] = 0, 0
                ndcgs[n], aps[n] = 0, 0
            
            test_batches = [0+i*batchSize for i in range(len(trainVector)) if 0+i*batchSize<len(trainVector)]
            test_batches_pbar = tqdm(test_batches, leave=False, desc='Evaluating ...')

            
            for batch in test_batches_pbar:
                leftIndex = batch
                rightIndex = leftIndex + batchSize if leftIndex+batchSize<len(trainVector) else len(trainVector)
                user_neighbors_ratings_batch = user_neighbors_ratings[leftIndex:rightIndex].to(device)
                data = Variable(copy.deepcopy(trainVector[leftIndex:rightIndex])).to(device)

                # Neighbor interacted module
                user_pref_vector = torch.tensor([]).to(device)
                for idx in range(leftIndex, rightIndex):
                    user_id = all_user_id[idx]
                    # 取出單一使用者對所有遊戲的喜好機率
                    user_logits = all_user_logits[user_id].to(device)  # [1, 428]
                    user_pref_vector = torch.cat([user_pref_vector, user_logits], 0)  # [b, 428]

                #  Exclude the purchased vector that have occurred in the training set
                fakeData_nbr = G_nbr(data, user_pref_vector, user_neighbors_ratings_batch)
                fakeData = fakeData_nbr
                result = fakeData + Variable(torch.where(copy.deepcopy(trainMaskVector[leftIndex:rightIndex].to(device))>0, -100, 0))
                all_user_top.append(result)

                # 計算 batch 的各項指標
                for n in top_k:
                    batch_precision, batch_recall, batch_ndcg, batch_ap = computeTopK(test_data[leftIndex:leftIndex + batchSize], 
                                                                                      result, n, item_id_map)
                    precisions[n] += batch_precision
                    recalls[n] += batch_recall
                    ndcgs[n] += batch_ndcg
                    aps[n] += batch_ap

            # 計算 total 的各項指標
            for n in top_k:
                precisions[n] = precisions[n] / n_user
                recalls[n] = recalls[n] / n_user
                ndcgs[n] = ndcgs[n] / n_user
                aps[n] = aps[n] / n_user

                result_precision[n] = np.concatenate((result_precision[n], np.array([[epoch, precisions[n]]])), axis = 0)
                result_recall[n] = np.concatenate((result_recall[n], np.array([[epoch, recalls[n]]])), axis = 0)
                result_ndcg[n] = np.concatenate((result_ndcg[n], np.array([[epoch, ndcgs[n]]])), axis = 0)
                result_map[n] = np.concatenate((result_map[n], np.array([[epoch, aps[n]]])), axis = 0)

            g_loss_list = np.concatenate((g_loss_list,np.array([[epoch,g_loss.item()]])), axis = 0)
            d_loss_list = np.concatenate((d_loss_list,np.array([[epoch,d_loss.item()]])), axis = 0)


            pbar.set_postfix(desc='d_loss: {:.6f}, g_loss: {:.6f}, precision: {:}, recall: {:}, ndcg: {:} , map: {:}'
                             .format(d_loss.item(), g_loss.item(),
                                     '{}'.format([ 'top'+str(n)+':'+str(round(precisions[n], 3)) for n in top_k]),
                                     '{}'.format([ 'top'+str(n)+':'+str(round(recalls[n], 3)) for n in top_k]),
                                     '{}'.format([ 'top'+str(n)+':'+str(round(ndcgs[n], 3)) for n in top_k]),
                                     '{}'.format([ 'top'+str(n)+':'+str(round(aps[n], 3)) for n in top_k]),))
            if ndcgs[config['top_k'][-1]] > top_ndcg:
                best_ep = epoch
                top_ndcg = ndcgs[config['top_k'][-1]]
                torch.save(G_nbr.state_dict(), 'model/parameters/RNGAN_G')
                torch.save(D.state_dict(), 'model/parameters/RNGAN_D')
                # torch.save(nbr_module.state_dict(), 'model/nbr_module_NeighborGAN')
            
            # 每 50 epoch 畫一次圖，先不用
            # if(epoch%50==0 and epoch>10):
            if epoch%1000 == 999:
                for n in top_k:
                    print('top',n,':','-'*30)
                    result_plt(result_precision[n][1:,], 'precision')
                    result_plt(result_recall[n][1:,], 'recall')
                    result_plt(result_ndcg[n][1:,], 'ndcg')
                    # result_plt(result_map[n][1:,])
                    result_plt(g_loss_list[1:,], 'Gloss')
                    result_plt(d_loss_list[1:,], 'Dloss')
                print('='*40)


    return result_precision, result_recall, result_ndcg, result_map, g_loss_list[1:,], d_loss_list[1:,], all_user_top, best_ep

def GAN_pred(add_neg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = 'data/preprocessed_data.pkl' if add_neg==False else 'data/Exp_data/preprocessed_data.pkl'
    data = pd.read_pickle(data_path)
    trainVector = data['trainVector'].to(device)
    all_user_logits = pd.read_pickle('data/User/all_user_logits.pkl')
    all_user_id = list(data['test_data'].keys())
    user_neighbors_ratings = pd.read_pickle('data/User/user_id_nbrRating.pkl')
    user_neighbors_ratings = torch.from_numpy(np.array(list(user_neighbors_ratings.values()))).to(device)
    G_nbr = Generator_nbr(data['itemCount'], num_heads=1).to(device)
    G_nbr.load_state_dict( torch.load('model/parameters/RNGAN_G') )

    leftIndex, rightIndex = 0, len(all_user_id)
    # Neighbor interacted module
    neighbor_based_rating_vector = torch.tensor([]).to(device)
    for idx in range(leftIndex, rightIndex):
        user_id = all_user_id[idx]
        # 取出單一使用者對所有遊戲的喜好機率
        user_logits = all_user_logits[user_id].to(device)  # [1, 428]
        neighbor_based_rating_vector = torch.cat([neighbor_based_rating_vector, user_logits], 0)  # [b, 428]

    #  Exclude the purchased vector that have occurred in the training set
    fakeData = G_nbr(trainVector, neighbor_based_rating_vector, user_neighbors_ratings)
    
    return fakeData.detach().cpu()
