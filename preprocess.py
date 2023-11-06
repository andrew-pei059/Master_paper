# 清理評論文字 & 篩選評論
import re
import torch
import faiss
import pickle
import random
import pandas as pd
import numpy as np
import torch.nn.functional as nn

from tqdm import tqdm
from sklearn.decomposition import NMF
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

# 刪除非英文的評論
def has_letter(text):
    return bool(re.search(r'[a-zA-Z]', text))

# 選擇類別
# select_cate 中，如果刪掉RPG，不管 threshold 怎麼設定都無法穩定。但不刪RPG 資料又太多 -> 目前有刪RPG
def select_app_category(df):
    select_cate = ['Indie', 'Action', 'Casual', 'Adventure', 'Simulation', 'Strategy', 'RPG']
    select_cate = select_cate[:-1]
    genre = pd.read_csv('data/raw_data/app_review_summary.csv')
    # game_cate(eg 'Action,Adventure,Strategy')，split 後檢查是否有任一類別在 select_cate 中
    app_list = [ genre['AppID'][i] for i,game_cates in enumerate(genre['Genres']) 
                if ([cate for cate in game_cates.split(',') if cate in select_cate] != []) ]
    df = df[ df['AppID'].isin(app_list) ]

    return df

def filter_user_app(df, app_threshold, user_threshold):
    # 篩選評論數
    app_reviews = df['AppID'].value_counts()
    df = df[ df['AppID'].isin(app_reviews[app_reviews >= app_threshold].index[:]) ]
    user_reviews = df['UserID'].value_counts()
    df = df[ df['UserID'].isin(user_reviews[user_reviews >= user_threshold].index[:]) ]
    df.reset_index(drop=True, inplace=True)

    return df

# --------------------------------------------------------------------

def clean_filter_data(df):
    df = df[ df['Review'].apply(has_letter) ]
    df.reset_index(drop=True, inplace=True)
    # 分句，每句話只少要有 3 個字
    all_review_sents = []
    min_words_sent = 3
    for review in df['Review']:
        review_sents = []  # 存每篇評論處理好的所有句子
        # 先用 \n、\r 等來分句
        sentences = review.splitlines()
        sentences = list(filter(None, sentences))
        # 再用符號 .!? 進行分句
        for sent in sentences:
            # [ <\[ ] 匹配 '<' or '[' | .*? 匹配任意字元 | [ >\] ] 匹配'>' or ']'
            sent = re.sub(r'[<\[].*?[>\]]', '', sent)  # 清除 HTML tag, ex: [h1]、<h1>
            sent = re.split(r' *[\.\?!]', sent)
            # 去除頭尾的空白、刪除字數過少的句子
            sent = [ s.strip() for s in sent if len(s.strip().split())>min_words_sent ]
            sent = list(filter(None, sent))
            review_sents += sent
        all_review_sents.append(review_sents)

    df['SplitReview'] = all_review_sents
    # 刪除[]、非英文的評論
    empty_list = [ i for i,x in enumerate(df['SplitReview']) if x ==[] ]
    df.drop(empty_list, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # --------------------------------------------------------------------

    # 接著是篩選評論數
    filter_data = select_app_category(df)
    # 反覆檢查使用者評論數、遊戲評論數是否平衡
    app_threshold, user_threshold, max_count = 40, 30, 32
    count, min_app = 0, 1
    while (min_app < app_threshold) and (count < max_count):
        filter_data = filter_user_app(filter_data, app_threshold, user_threshold)
        min_app = min(filter_data['AppID'].value_counts())
        count += 1
    
    filter_data['Like'] = filter_data['Like'].apply(lambda x : 0 if x == False else 1)
    filter_data['Like'] = filter_data['Like'].astype(int)
    filter_data['UserID'] = filter_data['UserID'].astype('int64')
    filter_data['RefValue'] = filter_data['RefValue'].astype(float)
    filter_data['VoteUp'] = filter_data['VoteUp'].astype(int)  # int64 -> int32
    # 之後會優先挑選 RefValue 高的評論
    # RefValue 最低 0.073。RefValue == 0 的資料最多有7個 Vote
    filter_data.loc[ filter_data['RefValue']==0, 'RefValue' ] += filter_data.loc[ filter_data['RefValue']==0, 'VoteUp' ]*0.01
    # 確保使用者正評數 15 以上，這樣 test 中至少有 3 個 item
    rating_matrix = filter_data.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0)
    user_ids = rating_matrix.index
    pos_reviews_threshold = 15  # 理想是 25，才能確保 test 中有 5 個，但 App 會只剩23評論
    del_users = [ user_ids[uid] for uid in np.where( np.sum(rating_matrix.values, axis=1)<pos_reviews_threshold )[0] ]
    trash_users = [76561197981638563, 76561198133519920, 76561197991874834, 76561198003977147, 76561198123305993, 76561198032835217 ]  # 看到的無用評論
    trash_users += del_users
    filter_data = filter_data[ ~(filter_data['UserID'].isin(trash_users)) ]
    filter_data.reset_index(drop=True, inplace=True)
    filter_data.to_pickle(f'data/reviews_{len(filter_data)}.pkl')

    # 輸出結果
    print('Apps: ', len(filter_data['AppID'].unique()), ' | Users: ', len(filter_data['UserID'].unique()), 'total data: ', len(filter_data))
    print('App reviews at least', min(filter_data['AppID'].value_counts()), '| User reviews at least', min(filter_data['UserID'].value_counts()))
    print( f'zero ratio: {filter_data.Like.value_counts().get(0, 0)*100/len(filter_data):.3f}')

    return filter_data

# --------------------------------------------------------------------
# use BERT to get review embedding
class RevDataSet(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews
    def __getitem__(self, idx):
        return self.reviews[idx]
    def __len__(self):
        return len(self.reviews)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_bert_embedding(filter_data):
    configuration = BertConfig()
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # BERT embedding
    all_rev = []
    for sentences in filter_data['SplitReview']:
        all_rev.append( "[SEP]".join(sentences) )

    rev_dataset = RevDataSet(all_rev)
    rev_loader = DataLoader(rev_dataset, batch_size=32, shuffle=False)

    review_emb_list = []
    # 句字中字數的 avg=130, std=200
    MAX_LENGTH = 500

    model.eval()
    for review_batch in tqdm(rev_loader):
        torch.cuda.empty_cache()
        rev_encode = bert_tokenizer.batch_encode_plus(
            review_batch,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            return_attention_mask = True,
            max_length = MAX_LENGTH,
            truncation = True,
            padding = "max_length",
            return_tensors = 'pt'
        )
        with torch.no_grad():
            rev_emb = model(rev_encode["input_ids"].to(device), rev_encode["attention_mask"].to(device))
        
        review_emb_list.extend( rev_emb[1].cpu().numpy() )

    filter_data['ReviewEmbedding'] = review_emb_list
    # save file
    filter_data.to_pickle(f'data/reviews_{len(filter_data)}.pkl')

# --------------------------------------------------------------------
# 前處理會用到的 function

# 訓練/測試 切分方法
def train_test_split(df, train_ratio=0.8):
    all_user_id = list(df.pivot_table(index='UserID', columns='AppID', values='Interacted').fillna(0).index)
    train_df, test_df = pd.DataFrame(), pd.DataFrame()  # 用於1對1 nbr module
    user_train_set, user_test_set = {}, {}  # 用於 GAN 的訓練資料、gt
    user_group = df.groupby('UserID')

    for uid in all_user_id:
        single_user_data = user_group.get_group(uid)
        # 打亂順序，避免 test 都取到相近的 App
        # single_user_data = single_user_data.sample(frac=1, random_state=69,ignore_index=True)
        # 如果全是 1 ，value_counts 無法用兩個變數接，會噴錯 !
        label_count = single_user_data.Like.value_counts()
        positive_count, neg_count = label_count.get(1, 0), label_count.get(0, 0)
        positive_data = single_user_data[ single_user_data.Like == 1 ]
        positive_data = positive_data.sample(frac=1, random_state=30, ignore_index=True)
        positive_data_train = positive_data[ : round(positive_count*train_ratio) ]  # int 會無條件捨去 -> train neg 較少
        positive_data_test = positive_data[ round(positive_count*train_ratio) : ]
        train_df = pd.concat([train_df, positive_data_train], axis=0)
        test_df = pd.concat([test_df, positive_data_test], axis=0)

        neg_data = single_user_data[ single_user_data.Like == 0 ]
        neg_data = neg_data.sample(frac=1, random_state=30, ignore_index=True)
        neg_data_train = neg_data[ : round(neg_count*train_ratio) ]
        neg_data_test = neg_data[ round(neg_count*train_ratio) : ]
        train_df = pd.concat([train_df, neg_data_train], axis=0)
        test_df = pd.concat([test_df, neg_data_test], axis=0)
        # train 中是互動過的 App(不論正負評)，test 中只有正評的 App
        single_train = pd.concat([positive_data_train, neg_data_train], axis=0)
        user_train_set[uid] = list(single_train.AppID.values[:])
        user_test_set[uid] = list(positive_data_test.AppID.values[:])

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df = train_df.sample(frac=1, random_state=25, ignore_index=True)
    # 顯示結果
    print(f'train: {len(train_df)}, neg ratio: {train_df.Like.value_counts().get(0)/len(train_df):.3f}, Like:{train_df.Like.value_counts().get(1)}')
    print(f'test: {len(test_df)}, neg ratio: {test_df.Like.value_counts().get(0)/len(test_df):.3f}')

    return train_df, test_df, user_train_set, user_test_set

def get_train_interaction_vector(matrix, user_train_set):
    app_ids = list(matrix.columns)
    user_ids = user_train_set.keys()
    # 以 train 中的資料重建 interaction matrix

    return [ [1 if app_id in user_train_set[user_id] else 0 for app_id in app_ids ] for user_id in user_ids ]

def matrix_factorization(matrix, trainVector):
    # n_components is embedding dimension。vervose=1 顯示訓練過程
    model = NMF(n_components=512, init='random', max_iter=200, random_state=25, verbose=0)
    user_embeddings = model.fit_transform(trainVector*matrix.values)
    # 轉成 float32 有兩種寫法
    encoded_user_embeddings = np.asarray(user_embeddings, dtype=np.float32)
    app_embeddings = model.components_.T
    encoded_app_embeddings = np.asarray(app_embeddings.astype('float32'))
    user_id_emb = dict( zip(matrix.index, encoded_user_embeddings) )
    app_id_emb = dict( zip(matrix.columns, encoded_app_embeddings) )

    return user_id_emb, app_id_emb

# 找 user_nbr 的 rating vector
def get_nbrRating(user_id_nbr, rating_matrix):
    nbr_rating_list = []
    for uid in user_id_nbr.keys():
        nbr = user_id_nbr[uid]
        nbr_rating = []
        for n_uid in nbr:
            nbr_rating.append(rating_matrix.loc[n_uid].values)
        nbr_rating_list.append( np.sum(nbr_rating, axis=0) )
    
    user_id_nbrRating = dict( zip(user_id_nbr.keys(), nbr_rating_list) )

    return user_id_nbrRating

# 找 user、app 的 k 個鄰居(id)。以及加總 user_nbr 的 rating vector 
def get_nbr(user_id_emb, app_id_emb, k=6):
    user_id_emb = pd.read_pickle('data/User/user_id_emb.pkl')
    app_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')

    user_emb = np.asarray( list(user_id_emb.values()) )
    user_embeddings_index = faiss.IndexIDMap(faiss.IndexFlatIP(user_emb.shape[1]))
    user_embeddings_index.add_with_ids( user_emb, np.array(list(user_id_emb.keys())) )
    app_emb = np.asarray( list(app_id_emb.values()) )
    app_embeddings_index = faiss.IndexIDMap(faiss.IndexFlatIP(app_emb.shape[1]))
    app_embeddings_index.add_with_ids( app_emb, np.array(list(app_id_emb.keys())) )

    # 找出每筆資料的 k neighbors。user_nbr_idx 是 list，每個元素存 k 個鄰居的 user_id
    user_nbr_distances, user_nbr_idx = user_embeddings_index.search(user_emb, k)
    app_nbr_distances, app_nbr_idx = app_embeddings_index.search(app_emb, k)
    # index 0 是自己，需要刪掉
    user_nbr_idx = np.delete(user_nbr_idx, 0, axis=1)
    app_nbr_idx = np.delete(app_nbr_idx, 0, axis=1)
    # user_id_nbr : { user_id : user_nbr_id }。user_nbr_id 的型態是 ndarray，shape=(882, 5)
    user_id_nbr = dict( zip(user_id_emb.keys(), user_nbr_idx) )
    app_id_nbr = dict( zip(app_id_emb.keys(), app_nbr_idx) )
    
    return user_id_nbr, app_id_nbr

# nonzero 函數回傳的是 index。搭配 id_map 取得 user_id、app_id
def get_user_interacted_apps(matrix, k=30):
    app_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')
    app_id_map = { idx:aid for idx,aid in enumerate(matrix.columns) }

    # user_interacted_apps_emb 以字典形式儲存 { user_id: interacted_app_emb }
    interacted_app_ids = []
    interacted_app_emb = []
    for uid in matrix.index:
        # 取出有評論之 app 的索引，再到 app_id_map 取出對應的 app_id
        app_idx = matrix.loc[uid].values.nonzero()[0]
        app_ids = [ app_id_map[idx] for idx in app_idx ]
        # 從 user 互動過的所有 apps 中隨機取 k 個
        app_ids = random.sample(app_ids, k)
        interacted_app_ids.append(app_ids)
        # 存 Embedding
        app_emb = [ app_id_emb[aid] for aid in app_ids ]
        interacted_app_emb.append(app_emb)

    user_interacted_appID = dict(zip( matrix.index, interacted_app_ids ))
    user_interacted_appEmb = dict(zip( matrix.index, interacted_app_emb ))

    return user_interacted_appID, user_interacted_appEmb

def get_app_interacted_users(matrix, k=30):
    user_id_emb = pd.read_pickle('data/User/user_id_emb.pkl')
    user_id_map = { idx:uid for idx,uid in enumerate(matrix.index) }

    # interacted_user_emb 以字典形式儲存 { app_id: interacted_user_emb }
    interacted_user_ids, interacted_user_emb = [], []
    for aid in matrix.columns:
        # 取出有評論之 user 的索引，再到 user_id_map 取出對應的 user_id
        user_idx = matrix[aid].values.nonzero()[0]
        user_ids = [ user_id_map[idx] for idx in user_idx ]
        # 從與 app 有互動的所有 users 中隨機取 k 個
        user_ids = random.sample(user_ids, k)
        interacted_user_ids.append(user_ids)
        # 存 Embedding
        user_emb = [ user_id_emb[uid] for uid in user_ids ]
        interacted_user_emb.append( user_emb )

    app_interacted_userID = dict( zip( matrix.columns, interacted_user_ids ) )
    app_interacted_usersEmb = dict( zip( matrix.columns, interacted_user_emb ) )

    return app_interacted_userID, app_interacted_usersEmb

# 考量 RefVale ，選較有價值的評論
# User 寫過的評論 >= 30 | App 擁有的評論 >= 33
def get_user_review_emb(df, select_reviews=30):
    user_group = df.groupby('UserID')
    select_reviews = min( select_reviews, min(df.UserID.value_counts().values) )  # 防錯，後面那串應該30以上
    user_apps = pd.read_pickle('data/User/user_interacted_appID.pkl')  # For all user ids
    reviews_emb = []
    for uid in user_apps.keys():
        temp = user_group.get_group(uid)[['ReviewEmbedding', 'RefValue']]
        temp.sort_values(ascending=False, by='RefValue', inplace=True)
        user_reviews_emb = []
        good_reviews = min(select_reviews, len(temp[ temp.RefValue>0 ]))
        user_reviews_emb += list(temp['ReviewEmbedding'].head(good_reviews).values[:])
        # 當有 RefValue 的評論數不足 30 個，需從其他評論中額外再抽
        if good_reviews < select_reviews:
            sample_reviews = select_reviews - good_reviews
            user_reviews_emb += random.sample( list(temp[ temp.RefValue==0 ]['ReviewEmbedding'].values[:]), sample_reviews )
        
        reviews_emb.append(user_reviews_emb)
    user_id_ReviewsEmb = dict( zip(user_apps.keys(), reviews_emb) )

    return user_id_ReviewsEmb

def get_app_review_emb(df, select_reviews=30):
    app_group = df.groupby('AppID')
    select_reviews = min( select_reviews, min(df.AppID.value_counts().values) )  # 防錯，後面那串應該30以上
    app_users = pd.read_pickle('data/App/app_interacted_userID.pkl')  #  For all item ids
    reviews_emb = []
    for uid in app_users.keys():
        temp = app_group.get_group(uid)[['ReviewEmbedding', 'RefValue']]
        temp.sort_values(ascending=False, by='RefValue', inplace=True)
        app_reviews_emb = []
        good_reviews = min(select_reviews, len(temp[ temp.RefValue>0 ]))
        app_reviews_emb += list(temp['ReviewEmbedding'].head(good_reviews).values[:])
        # 當有 RefValue 的評論數不足 30 個，需從其他評論中額外再抽
        if good_reviews < select_reviews:
            sample_reviews = select_reviews - good_reviews
            app_reviews_emb += random.sample( list(temp[ temp.RefValue==0 ]['ReviewEmbedding'].values[:]), sample_reviews )
        
        reviews_emb.append(app_reviews_emb)
    app_id_ReviewsEmb = dict( zip(app_users.keys(), reviews_emb) )

    return app_id_ReviewsEmb

# 取鄰居的所有評論 ( 5*30 )
def get_user_nbrReviewsEmb():
    user_nbr = pd.read_pickle('data/User/user_id_nbr.pkl')
    user_revEmb = pd.read_pickle('data/User/user_id_ReviewsEmb.pkl')

    all_user_nbr_revEmb = []
    for uid in user_nbr.keys():
        nbr_revEmb = []  # 所有鄰居評論
        nbr_ids = user_nbr[uid]
        for nuid in nbr_ids:
            nbr_revEmb += user_revEmb[nuid]
        all_user_nbr_revEmb.append(nbr_revEmb)

    user_id_nbrReviewsEmb = dict( zip(user_nbr.keys(), all_user_nbr_revEmb) )

    return user_id_nbrReviewsEmb

def get_item_nbrReviewsEmb():
    app_nbr = pd.read_pickle('data/App/app_id_nbr.pkl')
    app_revEmb = pd.read_pickle('data/App/app_id_ReviewsEmb.pkl')

    all_app_nbr_revEmb = []
    for aid in app_nbr.keys():
        nbr_revEmb = []  # 所有鄰居評論
        app_ids = app_nbr[aid]
        for naid in app_ids:
            nbr_revEmb += app_revEmb[naid]
        all_app_nbr_revEmb.append(nbr_revEmb)

    app_id_nbrReviewsEmb = dict( zip(app_nbr.keys(), all_app_nbr_revEmb) )

    return app_id_nbrReviewsEmb

# --------------------------------------------------------------------

# 前處理 (包含MF、找鄰居)
def preprocess_data(df):
    # construct rating matrix and interaction matrix
    df.drop(columns=['VoteUp'], inplace=True)
    df['Interacted'] = 1
    interaction_matrix = df.pivot_table(index='UserID', columns='AppID', values='Interacted').fillna(0)
    rating_matrix = df.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0)
    train_data, test_data, user_train_set, user_test_set = train_test_split(df, train_ratio=0.8)
    trainVector = torch.tensor( train_data.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0).values, dtype=torch.float )
    trainMaskVector = torch.tensor( get_train_interaction_vector(interaction_matrix, user_train_set), dtype=torch.float32 )  # 有評論過的都是 1
    user_id_emb, app_id_emb = matrix_factorization(rating_matrix, trainVector)
    # save file
    userCount, appCount = df.UserID.unique().shape[0], df.AppID.unique().shape[0]
    item_id_map = { aid:idx for idx,aid in enumerate( list(app_id_emb.keys()) ) }
    data = {
        'train_data': user_train_set,
        'trainVector': trainVector,
        'trainMaskVector': trainMaskVector,
        'test_data': user_test_set,
        'item_id_map': item_id_map,
        'userCount': userCount,
        'itemCount': appCount,
    }
    with open('data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open('data/User/user_id_emb.pkl', 'wb') as f:
        pickle.dump(user_id_emb, f)
    with open('data/App/app_id_emb.pkl', 'wb') as f:
        pickle.dump(app_id_emb, f)

    # 用 faiss 套件計算相似，找出 使用者/遊戲項目 的鄰居
    user_id_nbr, app_id_nbr = get_nbr(user_id_emb, app_id_emb, 6)
    nbr_rating_matrix = df.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0.5)
    user_id_nbrRating = get_nbrRating(user_id_nbr, nbr_rating_matrix)

    with open('data/User/user_id_nbr.pkl', 'wb') as f:
        pickle.dump(user_id_nbr, f)
    with open('data/App/app_id_nbr.pkl', 'wb') as f:
        pickle.dump(app_id_nbr, f)
    with open('data/User/user_id_nbrRating.pkl', 'wb') as f:
        pickle.dump(user_id_nbrRating, f)
    
    # 使用者互動過的遊戲個數、與遊戲有互動的使用者人數
    interacted_apps, interacted_users = 30, 30
    user_interacted_appID, user_interacted_appEmb = get_user_interacted_apps(interaction_matrix, interacted_apps)
    app_interacted_userID, app_interacted_userEmb = get_app_interacted_users(interaction_matrix, interacted_users)
    with open('data/User/user_interacted_appID.pkl', 'wb') as f:
        pickle.dump(user_interacted_appID, f)
    with open('data/App/app_interacted_userID.pkl', 'wb') as f:
        pickle.dump(app_interacted_userID, f)
    with open('data/User/user_interacted_appEmb.pkl', 'wb') as f:
        pickle.dump(user_interacted_appEmb, f)
    with open('data/App/app_interacted_userEmb.pkl', 'wb') as f:
        pickle.dump(app_interacted_userEmb, f)

    # 挑選評論來代表 使用者/遊戲項目 本身的特徵。考量 RefVale ，選較有價值的評論
    user_id_ReviewsEmb = get_user_review_emb(df)
    app_id_ReviewsEmb = get_app_review_emb(df)
    with open('data/User/user_id_ReviewsEmb.pkl', 'wb') as f:
        pickle.dump(user_id_ReviewsEmb, f)
    with open('data/App/app_id_ReviewsEmb.pkl', 'wb') as f:
        pickle.dump(app_id_ReviewsEmb, f)
    
    # 特定使用者其鄰居的所有評論
    user_id_nbrReviewsEmb = get_user_nbrReviewsEmb()
    app_id_nbrReviewsEmb = get_item_nbrReviewsEmb()
    with open('data/User/user_id_nbrReviewsEmb.pkl', 'wb') as f:
        pickle.dump(user_id_nbrReviewsEmb, f)
    with open('data/App/app_id_nbrReviewsEmb.pkl', 'wb') as f:
        pickle.dump(app_id_nbrReviewsEmb, f)

def add_neg_data(df, nbr_count):
    df = df[['AppID', 'UserID', 'Like', 'RefValue']]
    user_nbr = pd.read_pickle('data/User/user_id_nbr.pkl')
    user_group = df.groupby('UserID')
    all_apps, all_users = [], []
    for uid in user_nbr.keys():
        temp = user_group.get_group(uid)
        user_interacted_appID = list( temp['AppID'].values )
        # 鄰居討厭的 appID，用來判斷是否與使用者互動過的遊戲重複
        nbr_negative_appID = []
        for nuid in user_nbr[uid][:nbr_count]:  # 取5個鄰居，討厭的遊戲會變太多
            temp = user_group.get_group(nuid)
            nbr_negative_appID = nbr_negative_appID + list( temp[ temp['Like']==0 ]['AppID'].values )

        nbr_negative_appID = list(set(nbr_negative_appID))
        new_negative = [ aid for aid in nbr_negative_appID if aid not in user_interacted_appID ]
        all_apps = all_apps + new_negative
        all_users = all_users + [ uid for _ in range( len(new_negative) ) ]

    all_y = [ 0 for _ in range(len(all_apps)) ]
    d = pd.DataFrame(zip(all_apps, all_users, all_y), columns=['AppID', 'UserID', 'Like'])
    df = pd.concat([df, d], axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    # save file
    df.to_pickle(f'data/Exp_data/reviews_nbr{nbr_count}_{len(df)}.pkl')

    # 用新 dataset 建立 trainVector
    df['Interacted'] = 1
    interaction_matrix = df.pivot_table(index='UserID', columns='AppID', values='Interacted').fillna(0)
    rating_matrix = df.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0)
    train_data, test_data, user_train_set, user_test_set = train_test_split(df, train_ratio=0.8)
    trainVector = torch.tensor( train_data.pivot_table(index='UserID', columns='AppID', values='Like').fillna(0).values, dtype=torch.float )
    trainMaskVector = torch.tensor( get_train_interaction_vector(interaction_matrix, user_train_set), dtype=torch.float32 )  # 有評論過的都是 1
    user_id_emb, app_id_emb = matrix_factorization(rating_matrix, trainVector)
    # save file
    userCount, appCount = df.UserID.unique().shape[0], df.AppID.unique().shape[0]
    item_id_map = { aid:idx for idx,aid in enumerate( list(app_id_emb.keys()) ) }
    data = {
        'train_data': user_train_set,
        'trainVector': trainVector,
        'trainMaskVector': trainMaskVector,
        'test_data': user_test_set,
        'item_id_map': item_id_map,
        'userCount': userCount,
        'itemCount': appCount,
    }
    with open('data/Exp_data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('data/Exp_data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('data/Exp_data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open('data/User/user_id_emb.pkl', 'wb') as f:
        pickle.dump(user_id_emb, f)
    with open('data/App/app_id_emb.pkl', 'wb') as f:
        pickle.dump(app_id_emb, f)
