import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, isdir, join

def test(df_path):

    # 取得所有檔案與子目錄名稱
    files = listdir(df_path)

    # 以迴圈處理
    for f in files:
        # 產生檔案的絕對路徑
        fullpath = join(df_path, f)
        # 判斷 fullpath 是檔案還是目錄
        if isfile(fullpath):
            print("檔案：", f)
        elif isdir(fullpath):
            print("目錄：", f)


class TestDataSet(Dataset):
    def __init__(self, uid):
        item_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')
        item_ids = list(item_id_emb.keys())
        user_ids = [ uid for _ in range(len(item_ids)) ]
        self.uid = uid
        self.temp = pd.DataFrame()
        self.temp['UserID'], self.temp['ItemID'] = user_ids, item_ids
        # User
        self.user_id_emb = pd.read_pickle('data/User/user_id_emb.pkl')
        self.user_interacted_appEmb = pd.read_pickle('data/User/user_interacted_appEmb.pkl')
        self.user_id_nbrReviewsEmb = pd.read_pickle('data/User/user_id_nbrReviewsEmb.pkl')
        self.user_id_ReviewsEmb = pd.read_pickle('data/User/user_id_ReviewsEmb.pkl')
        # Item
        self.item_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')
        self.item_interacted_userEmb = pd.read_pickle('data/App/app_interacted_userEmb.pkl')
        self.item_id_nbrReviews_emb = pd.read_pickle('data/App/app_id_nbrReviewsEmb.pkl')
        self.item_id_ReviewsEmb = pd.read_pickle('data/App/app_id_ReviewsEmb.pkl')

    def __getitem__(self, idx):
        user_id = self.uid
        item_id = self.temp['ItemID'][idx]
        # user side
        # 會在 user_nbr_module 用到的資料
        user_data = {}
        user_emb = torch.tensor(self.user_id_emb[user_id], dtype=torch.float32)
        # 因為最外面是 list，所以要先轉成 ndarray
        user_itemEmb = torch.tensor( np.array(self.user_interacted_appEmb[user_id]), dtype=torch.float32 )
        user_nbr_revEmb = torch.tensor( np.array(self.user_id_nbrReviewsEmb[user_id]), dtype=torch.float32 )
        user_revEmb = torch.tensor( np.array(self.user_id_ReviewsEmb[user_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        user_data['user_id'], user_data['user_emb'], user_data['user_itemEmb'] = user_id, user_emb, user_itemEmb
        user_data['user_nbr_revEmb'], user_data['user_revEmb'] = user_nbr_revEmb, user_revEmb
        # item side
        # 會在 item_nbr_module 用到的資料
        item_data = {}
        item_emb = torch.tensor(self.item_id_emb[item_id], dtype=torch.float32)
        item_userEmb = torch.tensor( np.array(self.item_interacted_userEmb[item_id]), dtype=torch.float32 )
        item_nbr_revEmb = torch.tensor( np.array(self.item_id_nbrReviews_emb[item_id]), dtype=torch.float32 )
        item_revEmb = torch.tensor( np.array(self.item_id_ReviewsEmb[item_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        item_data['item_id'], item_data['item_emb'], item_data['item_userEmb'] = item_id, item_emb, item_userEmb
        item_data['item_nbr_revEmb'], item_data['item_revEmb'] = item_nbr_revEmb, item_revEmb

        return user_data, item_data

    def __len__(self):
        return len(self.temp)


class ReviewDataset(Dataset):
    def __init__(self, target, df_path):
        if target =="train":
            self.review_df = pd.read_pickle(df_path)
        elif target =="test":
            self.review_df = pd.read_pickle(df_path)
        # User
        self.user_id_emb = pd.read_pickle('data/User/user_id_emb.pkl')
        self.user_interacted_appEmb = pd.read_pickle('data/User/user_interacted_appEmb.pkl')
        self.user_id_nbrReviewsEmb = pd.read_pickle('data/User/user_id_nbrReviewsEmb.pkl')
        self.user_id_ReviewsEmb = pd.read_pickle('data/User/user_id_ReviewsEmb.pkl')
        # Item
        self.item_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')
        self.item_interacted_userEmb = pd.read_pickle('data/App/app_interacted_userEmb.pkl')
        self.item_id_nbrReviews_emb = pd.read_pickle('data/App/app_id_nbrReviewsEmb.pkl')
        self.item_id_ReviewsEmb = pd.read_pickle('data/App/app_id_ReviewsEmb.pkl')
      
    def __getitem__(self, idx):
        user_id = self.review_df["UserID"][idx]
        item_id = self.review_df["AppID"][idx]
        y = self.review_df["Like"][idx]

        # user side
        # 會在 user_nbr_module 用到的資料
        user_data = {}
        user_emb = torch.tensor(self.user_id_emb[user_id], dtype=torch.float32)
        # 因為最外面是 list，所以要先轉成 ndarray
        user_itemEmb = torch.tensor( np.array(self.user_interacted_appEmb[user_id]), dtype=torch.float32 )
        user_nbr_revEmb = torch.tensor( np.array(self.user_id_nbrReviewsEmb[user_id]), dtype=torch.float32 )
        user_revEmb = torch.tensor( np.array(self.user_id_ReviewsEmb[user_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        user_data['user_id'], user_data['user_emb'], user_data['user_itemEmb'] = user_id, user_emb, user_itemEmb
        user_data['user_nbr_revEmb'], user_data['user_revEmb'] = user_nbr_revEmb, user_revEmb

        # item side
        # 會在 item_nbr_module 用到的資料
        item_data = {}
        item_emb = torch.tensor(self.item_id_emb[item_id], dtype=torch.float32)
        item_userEmb = torch.tensor( np.array(self.item_interacted_userEmb[item_id]), dtype=torch.float32 )
        item_nbr_revEmb = torch.tensor( np.array(self.item_id_nbrReviews_emb[item_id]), dtype=torch.float32 )
        item_revEmb = torch.tensor( np.array(self.item_id_ReviewsEmb[item_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        item_data['item_id'], item_data['item_emb'], item_data['item_userEmb'] = item_id, item_emb, item_userEmb
        item_data['item_nbr_revEmb'], item_data['item_revEmb'] = item_nbr_revEmb, item_revEmb

        # --------------------------------------------
        bce_y = torch.zeros(2, dtype=torch.float32)
        bce_y[y] = 1.0
        # -------------------------------------------
        
        return user_data, item_data, bce_y, y

    def __len__(self):
        return len(self.review_df)