import os
from math import sqrt

import pandas as pd
from Rec.base_models.MetricsEvaluation import RecEval
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data(path):
    # 读取数据
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(path, 'ratings.dat'), sep='::', engine='python', names=column_names)
    # 分割训练集和测试集
    train_data, test_data = train_test_split(ratings, test_size=0.2)
    # 将训练集和测试集按 user_id 分组后将 movie_id 以列表的形式存储并重新生成索引
    train_data = train_data.groupby('user_id')['movie_id'].apply(list).reset_index()
    test_data = test_data.groupby('user_id')['movie_id'].apply(list).reset_index()
    # 将训练集和测试集转化成字典的形式存储：{user_id: [item_id1, item_id2,...,item_idn]}
    train_user_items = {}
    for user, movies in zip(*(list(train_data['user_id']), list(train_data['movie_id']))):
        train_user_items[user] = set(movies)
    test_user_items = {}
    for user, movies in zip(*(list(test_data['user_id']), list(test_data['movie_id']))):
        test_user_items[user] = set(movies)
    return train_user_items, test_user_items


def Item_CF_Rec(train_user_items, test_user_items, K, N):
    """
    基于物品的协同过滤编程实现
    :param train_user_items:训练集，数据格式：{user_id: [item_id1, item_id2,...,item_idn]}
    :param test_user_items:测试集，数据格式：{user_id: [item_id1, item_id2,...,item_idn]}
    :param K:表示与每个用户存在交互的每个物品最相似的 Top-K 个物品
    :param N:表示为当前用户推荐相似度最大的 Top-N 个物品
    :return:
    """
    # 1.建立 user -> item 的倒排索引
    # 由于训练集的数据格式已经符合，因此不用建立

    # 2.计算物品协同过滤矩阵
    # 利用 1. 得到的倒排索引统计物品与物品之间与共同的用户存在交互的次数
    # 物品协同过滤矩阵的数据格式：sim = {item_id1:{item_id2:num1}...}
    # 即使用嵌套字典表示物品与物品之间与共同用户存在交互的次数
    # 同时记录每个物品与多少用户存在交互，即 mp = {item_id1:num1, item_id2:num2...}
    sim = {}
    mp = {}
    for user_id, items in tqdm(train_user_items.items()):
        for i in items:
            if i not in mp:
                mp[i] = 0
            mp[i] += 1
            if i not in sim:
                sim[i] = {}
            for j in items:
                if j not in sim[i]:
                    sim[i][j] = 0
                if i != j:
                    sim[i][j] += 1

    # 3.计算物品相似度矩阵
    # 2. 得到的物品协同过滤矩阵相当于余弦相似度的分子部分
    # 分母部分则是物品间存在交互的用户数量乘积
    for i, items in tqdm(sim.items()):
        for j, score in items.items():
            if i != j:
                sim[i][j] = score / sqrt(mp[i] * mp[j])

    # 4.通过测试集为每个用户推荐相似度最高的前 N 个物品
    # 首先需要通过物品协同过滤矩阵得到与当前用户存在交互的物品最相似的前 K 个物品
    # 然后根据这 K 个物品，并从其中排除与测试集用户存在交互的物品，计算相似度分数
    # 计算结果是多个相似物品对当前物品的累加和
    items_rank = {}
    for uid, _ in tqdm(test_user_items.items()):  # 遍历测试集的每个用户
        items_rank[uid] = {}  # 存储当前用户的候选推荐物品
        for history_item in train_user_items[uid]:  # 遍历当前用户的历史喜欢物品，用于寻找与之相似的物品
            for item, score in sorted(sim[history_item].items(), key=lambda x: x[1], reverse=True)[:K]:
                if item not in train_user_items[uid]:  # 被推荐的物品不能在用户的历史喜欢物品里出现
                    if item not in items_rank[uid]:
                        items_rank[uid][item] = 0
                    items_rank[uid][item] += score

    items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in items_rank.items()}
    items_rank = {k: set([x[0] for x in v]) for k, v in items_rank.items()}

    return items_rank


if __name__ == '__main__':
    root_path = '../data/ml-1m/'
    train_user_items, test_user_items = load_data(root_path)
    rec_items = Item_CF_Rec(train_user_items, test_user_items, 80, 10)
    RecEval.rec_eval(train_user_items, test_user_items, rec_items)
