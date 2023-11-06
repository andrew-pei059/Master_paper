import torch
import torch.nn as nn
import pickle
import pandas as pd
from tqdm import tqdm, trange
from model.review_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, confusion_matrix


# Tanh 被限制在 +- 1，
class DimensionReduction(nn.Module):
    def __init__(self, ini_rev_dim=768, emb_dim=512):
        super(DimensionReduction, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(ini_rev_dim, emb_dim),
            # nn.Tanh(),
            # nn.Linear(256, 512),
            # nn.Tanh()
        )

    def forward(self, batch):
        dim_output = self.seq(batch)
        
        return dim_output
    
class ComponentAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(ComponentAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, batch):
        # batch.shape => (batch, seq_len, embed_dim)
        attn_output, attn_output_weights = self.multihead_attn(batch, batch, batch)
        
        return attn_output
    
class Coattention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(Coattention, self).__init__()
        self.multihead_attn_1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.multihead_attn_2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, rev_batch, batch_2):
        # batch_2 may be reviewEmb or user/item embedding
        attn_output, attn_output_weights = self.multihead_attn_1(rev_batch, batch_2, batch_2)  # rev_batch shape
        rev_batch_output = torch.mean(attn_output, 1)
        
        attn_output, attn_output_weights = self.multihead_attn_2(batch_2, rev_batch, rev_batch)  # batch_2 shape
        batch_2_output = torch.mean(attn_output, 1)

        return rev_batch_output, batch_2_output
    
class Neighbor_interacted_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(Neighbor_interacted_module, self).__init__()
        # DimensionReduction 將維度從 ini_rev_dim 變成 emb_dim
        self.rev_batch_rd = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.rev_batch_attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.batch_2_attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.co_attn = Coattention(emb_dim=emb_dim, num_heads=num_heads)
        
    def forward(self, rev_batch, batch_2):
        rev_batch = self.rev_batch_rd(rev_batch)
        rev_batch_attn = self.rev_batch_attn(rev_batch)
        batch_2_attn = self.batch_2_attn(batch_2)
        nbr_review_vector, latent_vector = self.co_attn(rev_batch_attn, batch_2_attn)
        
        return nbr_review_vector, latent_vector
    
class User_Item_interacted_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(User_Item_interacted_module, self).__init__()
        self.rd_1 = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.rd_2 = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.co_attn = Coattention(emb_dim=emb_dim, num_heads=num_heads)
        
    def forward(self, user_rev_batch, item_rev_batch):
        user_rev_batch = self.rd_1(user_rev_batch)
        item_rev_batch = self.rd_2(item_rev_batch)
        user_rev_vector, item_rev_vector = self.co_attn(user_rev_batch, item_rev_batch)
        
        return user_rev_vector, item_rev_vector
    
class Integrated_Neighbor_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(Integrated_Neighbor_module, self).__init__()
        self.user_nbr = Neighbor_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.review_interacted = User_Item_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.item_nbr = Neighbor_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.seq = nn.Sequential(
            nn.Linear( emb_dim+emb_dim, 512 ), # 串接
            nn.Dropout(0.3),
            nn.Tanh(),
            # nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, user_emb, user_itemEmb, user_nbr_revEmb, user_revEmb, item_revEmb, item_emb, item_nbr_revEmb, item_userEmb):
        # user/item side nbr interaction module 
        user_nbr_rev_vector, user_item_vector = self.user_nbr(user_nbr_revEmb, user_itemEmb)
        user_rev_vector, item_rev_vector = self.review_interacted(user_revEmb, item_revEmb)
        item_nbr_rev_vector, item_user_vector = self.item_nbr(item_nbr_revEmb, item_userEmb)

        # merge user side vectors
        user_nbr_rev_vector, user_item_vector = user_nbr_rev_vector.unsqueeze(1), user_item_vector.unsqueeze(1)
        user_emb, user_rev_vector = user_emb.unsqueeze(1), user_rev_vector.unsqueeze(1)
        user_nbr_interaction_vector = torch.cat( [user_nbr_rev_vector, user_item_vector, user_emb, user_rev_vector], 1 )
        # 用 attention 對 4 個 user 相關的向量做 reweight
        user_nbr_interaction_vector = self.attn(user_nbr_interaction_vector)
        user_nbr_interaction_vector = torch.mean(user_nbr_interaction_vector, 1)

        # merge item side vectors
        item_nbr_rev_vector, item_user_vector = item_nbr_rev_vector.unsqueeze(1), item_user_vector.unsqueeze(1)
        item_emb, item_rev_vector = item_emb.unsqueeze(1), item_rev_vector.unsqueeze(1)
        item_nbr_interaction_vector = torch.cat( [item_nbr_rev_vector, item_user_vector, item_emb, item_rev_vector], 1 )
        # 用 attention 對 4 個 item 相關的向量做 reweight
        item_nbr_interaction_vector = self.attn(item_nbr_interaction_vector)
        item_nbr_interaction_vector = torch.mean(item_nbr_interaction_vector, 1)

        # MLP ----------------------------
        fc_input = torch.cat( [user_nbr_interaction_vector, item_nbr_interaction_vector], 1 )
        fc_output = self.seq(fc_input)

        return fc_output

def Nbr_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df_path = 'data/train_data.pkl' if args['add_neg']==False else 'data/Exp_data/train_data.pkl'
    test_df_path = 'data/test_data.pkl' if args['add_neg']==False else 'data/Exp_data/test_data.pkl'
    train_dataset = ReviewDataset(target='train', df_path=train_df_path)
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False)
    test_dataset = ReviewDataset(target='test', df_path=test_df_path)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)
    # Initialize Model
    ini_rev_dim, emb_dim, num_heads = args['ini_rev_dim'], args['emb_dim'], args['num_heads']
    torch.manual_seed(args['seed'])  # 相同 seed 的話，each epoch 結果(loss、f1...)會相同
    nbr_module = Integrated_Neighbor_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads).to(device)

    # Loss criteria
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(nbr_module.parameters(), lr=args['lr'], weight_decay=1e-5)

    pbar = trange(args['epochCount'])
    train_loss_list, train_acc_list, train_precision_list, train_recall_list, train_f1_list = [], [], [], [], []
    test_loss_list, test_acc_list, test_precision_list, test_recall_list, test_f1_list = [], [], [], [], []
    best_ep, best_loss, best_f1 = 0, 1, 0.6

    for epoch in pbar:
        nbr_module.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 = [], [], [], [], []
        train_precision, train_recall = [], []

        for batch in train_loader:
            user_data, item_data, _, y = batch
            y = y.to(device)
            user_emb, user_itemEmb = user_data['user_emb'], user_data['user_itemEmb']
            user_revEmb, user_nbr_revEmb = user_data['user_revEmb'], user_data['user_nbr_revEmb']

            item_emb, item_userEmb = item_data['item_emb'], item_data['item_userEmb']
            item_revEmb, item_nbr_revEmb = item_data['item_revEmb'], item_data['item_nbr_revEmb']
            
            output_logits = nbr_module(user_emb.to(device), user_itemEmb.to(device), user_nbr_revEmb.to(device),
                                    user_revEmb.to(device), item_revEmb.to(device), item_emb.to(device),
                                    item_nbr_revEmb.to(device), item_userEmb.to(device) )
            
            # train model
            loss = criterion( torch.squeeze(output_logits, dim=1), y.float() )
            # Gradients stored in the parameters in the previous step should be cleared out first
            optimizer.zero_grad()
            # Compute the gradients for parameters
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(nbr_module.parameters(), max_norm=10)  試試有無影響
            # Update the parameters with computed gradients
            optimizer.step()
            result_logits = torch.where( output_logits > args['train_threshold'], 1, 0 ).squeeze(dim=1)
            acc = (result_logits == y).float().mean()
            precision = precision_score(y.cpu(), result_logits.cpu(), zero_division=0)
            recall = recall_score(y.cpu(), result_logits.cpu(), zero_division=0)
            f1 = f1_score(y.cpu(), result_logits.cpu(), zero_division=0)

            train_loss.append(loss.item())
            train_acc.append(acc.cpu())
            train_precision.append(precision)
            train_recall.append(recall)
            train_f1.append(f1)

        train_loss_list.append( np.mean(train_loss) )
        train_acc_list.append( np.mean(train_acc) )
        train_precision_list.append( np.mean(train_precision) )
        train_recall_list.append( np.mean(train_recall) )
        train_f1_list.append( np.mean(train_f1) )

        # ---------- Validation ----------
        nbr_module.eval()
        test_loss, test_acc, test_f1 = [], [], []
        test_precision, test_recall = [], []

        for batch in test_loader:
            user_data, item_data, _, y = batch
            y = y.to(device)
            user_emb, user_itemEmb = user_data['user_emb'], user_data['user_itemEmb']
            user_revEmb, user_nbr_revEmb = user_data['user_revEmb'], user_data['user_nbr_revEmb']

            item_emb, item_userEmb = item_data['item_emb'], item_data['item_userEmb']
            item_revEmb, item_nbr_revEmb = item_data['item_revEmb'], item_data['item_nbr_revEmb']
            
            with torch.no_grad():
                output_logits = nbr_module(user_emb.to(device), user_itemEmb.to(device), user_nbr_revEmb.to(device),
                                        user_revEmb.to(device), item_revEmb.to(device), item_emb.to(device),
                                        item_nbr_revEmb.to(device), item_userEmb.to(device) )
                
                loss = criterion( torch.squeeze(output_logits, dim=1), y.float() )
                result_logits = torch.where( output_logits > args['test_threshold'], 1, 0 ).squeeze(dim=1)
                acc = (result_logits == y).float().mean()
                precision = precision_score(y.cpu(), result_logits.cpu(), zero_division=0)
                recall = recall_score(y.cpu(), result_logits.cpu(), zero_division=0)
                f1 = f1_score(y.cpu(), result_logits.cpu(), zero_division=0)

                test_loss.append(loss.item())
                test_acc.append(acc.cpu())
                test_precision.append(precision)
                test_recall.append(recall)
                test_f1.append(f1)
        
        # save best model parameters
        cur_loss, cur_f1 = np.mean(test_loss), np.mean(test_f1)
        if (epoch>5) and ( cur_f1 > best_f1 ) and ( cur_loss < best_loss ):
            best_ep, best_loss, best_f1 = epoch, cur_loss, cur_f1
            torch.save(nbr_module.state_dict(), 'model/parameters/temp_best')

        test_loss_list.append(np.mean(test_loss))
        test_acc_list.append(np.mean(test_acc))
        test_precision_list.append(np.mean(test_precision))
        test_recall_list.append(np.mean(test_recall))
        test_f1_list.append(np.mean(test_f1))
        
        pbar.set_postfix( result= f"loss: {np.mean(test_loss):.5f}, acc: {np.mean(test_acc):.3f}\
            precision: {np.mean(test_precision):.3f}, recall: {np.mean(test_recall):.3f}, f1: {np.mean(test_f1):.3f}, save: {best_ep}" )
    
    print(f'best loss: {best_loss:.4f}, best f1:{best_f1:.3f}')

def Nbr_pred(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nbr_module = Integrated_Neighbor_module(args['ini_rev_dim'], args['emb_dim'], args['num_heads']).to(device)
    nbr_module.load_state_dict( torch.load('model/parameters/temp_best') )
    data_path = 'data/preprocessed_data.pkl' if args['add_neg']==False else 'data/Exp_data/preprocessed_data.pkl'
    data = pd.read_pickle(data_path)
    all_user_id = list(data['test_data'].keys())
    all_user_logits = {}
    for user_id in tqdm(all_user_id):
        temp_dataset = TestDataSet(uid=user_id)
        temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)
        user_logits = torch.tensor([])
        # 計算單一使用者對每個商品的喜好機率
        for temp_batch in temp_loader:
            user_data, item_data = temp_batch
            user_emb, user_itemEmb = user_data['user_emb'], user_data['user_itemEmb']
            user_revEmb, user_nbr_revEmb = user_data['user_revEmb'], user_data['user_nbr_revEmb']

            item_emb, item_userEmb = item_data['item_emb'], item_data['item_userEmb']
            item_revEmb, item_nbr_revEmb = item_data['item_revEmb'], item_data['item_nbr_revEmb']
            
            output_logits = nbr_module(user_emb.to(device), user_itemEmb.to(device), user_nbr_revEmb.to(device),
                                    user_revEmb.to(device), item_revEmb.to(device), item_emb.to(device),
                                    item_nbr_revEmb.to(device), item_userEmb.to(device) )
            
            output_logits = output_logits.detach().permute(1,0).cpu()  # [1, b]
            user_logits = torch.cat([user_logits, output_logits], 1)

        all_user_logits[user_id] = user_logits

    with open('data/User/all_user_logits.pkl', 'wb') as f:
        pickle.dump(all_user_logits, f)
