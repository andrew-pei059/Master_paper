import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

def select_negative_items(realData, itemCount, num_pm, num_zr):

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

def computeTopN(groundTruth, result, topN, item_id_map):
    # 因為使用者互動的商品最少只有 30 個，test 取0.2的話就只有6個 -> 算 Top 10 沒意義
    
    batch_precision, batch_recall = 0, 0
    batch_ndcg, batch_ap, batch_hit_ratio = 0, 0, 0

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

        for j in range(topN):
            if(res[j][1] in gt):
                hit += 1
                dcg += 1 / math.log2(j+2)
                sum_precs += hit / (j+1)

            if(idcgCount>0):
                idcg += 1/math.log2(j+2)
                idcgCount -= 1

        if len(gt)!=0 and topN!=0:
            precision = hit / topN
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
        if hit > 0 : batch_hit_ratio += 1
    
    return batch_precision, batch_recall, batch_ndcg, batch_ap, batch_hit_ratio

class Discriminator(nn.Module):
    def __init__(self, itemCount):
        super(Discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount, 125),
            nn.ReLU(True),
            nn.Linear(125, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        D_input = data
        result = self.dis(D_input)
        
        return result  
    
class Generator(nn.Module):
    def __init__(self, itemCount):
        self.itemCount = itemCount
        super(Generator, self).__init__()
        self.gen=nn.Sequential(
            # CFGAN
            nn.Linear(itemCount, 400),
            nn.ReLU(True),
            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, itemCount),
            nn.Sigmoid()
        )
    def forward(self, noise):
        # CFGAN
        G_input = noise
        result=self.gen(G_input)
    
        return result

def CFGAN_train(config):

    data = pd.read_pickle('data/preprocessed_data.pkl')
    userCount, itemCount, item_id_map = config['user_num'], data['itemCount'], data['item_id_map']
    trainVector, trainMaskVector, testSet = data['trainVector'], data['trainMaskVector'].clone(), list(data['test_data'].values())
    trainMaskVector = torch.where(trainMaskVector==1, -100, 0)
    # config
    epochCount, topN, alpha = config['epochCount'], config['top_k'], config['alpha']
    pro_PM, pro_ZR = int(config['pro_PM']*itemCount), int(config['pro_ZR']*itemCount)
    top_precision, g_seed, b_seed = 0, 49, 25

    result_precision, result_recall, result_ndcg, result_map, result_hr = {}, {}, {}, {}, {}
    for n in topN:
        result_precision[n], result_recall[n], result_ndcg[n], result_map[n], result_hr[n] = np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2))

    # Build the generator and discriminator
    torch.manual_seed(g_seed)
    G = Generator(itemCount)
    torch.manual_seed(g_seed)
    D = Discriminator(itemCount)
    regularization = nn.MSELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)
    G_step, D_step = 5, 2
    batchSize, batchSize_G, batchSize_D = 64, 32, 32

    pbar = tqdm(range(epochCount))
    for epoch in pbar:
        
        # ---------------------
        #  Train Generator
        # ---------------------
        b_seed += 3
        random.seed(b_seed)
        for step in range(G_step):
            
            # Select a random batch of purchased vector
            leftIndex = random.randint(0, userCount-batchSize_G-1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            
            # Select a random batch of negative items for every user
            n_items_pm, n_items_zr = select_negative_items(realData, itemCount, pro_PM, pro_ZR)           
            ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp
            
            # Generate a batch of new purchased vector
            fakeData = G(realData) 
            fakeData_ZP = fakeData * (eu + ku_zp)  
            fakeData_result = D(fakeData_ZP) 
            
            # Train the discriminator
            g_loss = np.mean(np.log(1.-fakeData_result.detach().numpy()+10e-5))  + alpha*regularization(fakeData_ZP,realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            

        # ---------------------
        #  Train Discriminator
        # ---------------------
        b_seed += 1
        random.seed(b_seed)
        for step in range(D_step):

            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount-batchSize_D-1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex+batchSize_D])) 
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            
            # Select a random batch of negative items for every user
            n_items_pm, _ = select_negative_items(realData, itemCount, pro_PM, pro_ZR)
            ku = Variable(torch.tensor(n_items_pm))
            
            # Generate a batch of new purchased vector
            fakeData = G(realData) 
            fakeData_ZP = fakeData * (eu + ku)
            
            # Train the discriminator
            fakeData_result = D(fakeData_ZP) 
            realData_result = D(realData) 
            d_loss = -np.mean(np.log(realData_result.detach().numpy()+10e-5) + 
                                np.log(1. - fakeData_result.detach().numpy()+10e-5)) + 0*regularization(fakeData_ZP,realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
        

        # ---------------------
        #  Evaluation
        # ---------------------
        if(epoch%10 == 9):

            n_user = len(testSet)
            index = 0
            precisions, recalls, ndcgs, aps, hrs = {}, {}, {}, {}, {}
            for n in topN:
                precisions[n], recalls[n], ndcgs[n], aps[n], hrs[n] = 0, 0, 0, 0, 0

            test_batches = [0+i*batchSize for i in range(len(trainVector)) if 0+i*batchSize<len(trainVector)]
            # test_batches_pbar = tqdm(test_batches, leave=False, desc='Evaluating ...')
            for batch in  test_batches:
                leftIndex = batch
                data = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize]))
                result = G(data) + Variable(copy.deepcopy(trainMaskVector[leftIndex:leftIndex + batchSize]))
                gt = [ testSet[i] for i in range(leftIndex, leftIndex + batchSize) if i < len(testSet)]
                
                for tn in topN:
                    precision, recall, ndcg, ap, hr = computeTopN(gt, result, tn, item_id_map)
                    precisions[tn] += precision
                    recalls[tn] += recall
                    ndcgs[tn] += ndcg
                    aps[tn] += ap
                    hrs[tn] += hr
                
                index+=1

            for n in topN:
                precisions[n] = precisions[n]/n_user
                recalls[n] = recalls[n]/n_user
                ndcgs[n] = ndcgs[n]/n_user
                aps[n] = aps[n]/n_user
                hrs[n] = hrs[n]/n_user

                result_precision[n] = np.concatenate((result_precision[n], np.array([[epoch, precisions[n]]])), axis = 0)
                result_recall[n] = np.concatenate((result_recall[n], np.array([[epoch, recalls[n]]])), axis = 0)
                result_ndcg[n] = np.concatenate((result_ndcg[n], np.array([[epoch, ndcgs[n]]])), axis = 0)
                result_map[n] = np.concatenate((result_map[n], np.array([[epoch, aps[n]]])), axis = 0)
                result_hr[n] = np.concatenate((result_hr[n], np.array([[epoch, hrs[n]]])), axis = 0)
            
            pbar.set_postfix(desc=f'precision:{precisions[10]:.3f}, recall:{recalls[10]:.3f}, ndcg:{ndcgs[10]:.3f}, hit ratio:{hrs[10]:.3f}', refresh=True)
            
            if precisions[10] > top_precision:
                top_precision, best_ep = precisions[10], epoch
                torch.save(G.state_dict(), 'model/parameters/G_CFGAN')
                torch.save(D.state_dict(), 'model/parameters/D_CFGAN')

    return best_ep, result_precision, result_recall, result_ndcg, result_map, result_hr

def CFGAN_pred(config):
    
    data = pd.read_pickle('data/preprocessed_data.pkl')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainVector = data['trainVector'].to(device)
    G = Generator(data['itemCount']).to(device)
    G.load_state_dict( torch.load('model/parameters/G_CFGAN') )
    fakeData = G(trainVector).to(device)

    return fakeData.detach().cpu()
