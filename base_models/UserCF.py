import os.path
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


def User_CF_Rec(train_user_items, test_user_items, K, N):
    """
    基于用户的协同过滤实现
    :param train_user_items:训练集，数据格式：{user_id: [item_id1, item_id2,...,item_idn]}
    :param test_user_items:测试集，数据格式：{user_id: [item_id1, item_id2,...,item_idn]}
    :param K 表示为每个用户计算与其最相似的 K 个用户
    :param N 表示为每个用户推荐相似度最大的 N 个物品
    :return 计算与每个用户最相似的 Top-K 个用户推荐的相似度最大的 N 个物品
    """
    # 1.通过训练集创建 item -> user 的倒排索引
    # 即格式为 {item_id1:{user_id1, user_id2, ... , user_idn}, item_id2: ...} 表示每个物品被哪些用户有过点击或评分
    item_users = {}
    for user_id, items in tqdm(train_user_items.items()):
        for item in items:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(user_id)

    # 2.计算用户协同过滤矩阵
    # 利用 1. 生成的倒排索引统计用户与用户间的相同物品的交互量
    # 用户协同过滤矩阵的格式为 sim = {user_id1:{user_id2:item1}}...
    # 即使用字典嵌套表示用户与用户间存在共同交互的物品数量
    # 同时记录每个用户存在交互的物品数量，即 mp = {user_id1：num1, user_id2:num2, ...}
    sim = dict()
    mp = dict()
    for item, users in tqdm(item_users.items()):
        for u in users:
            if u not in mp:
                mp[u] = 0
            mp[u] += 1
            if u not in sim:
                sim[u] = {}
            for v in users:
                if u != v:
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1

    # 3.计算用户相似度矩阵
    # 2. 得到的用户协同过滤矩阵相当于余弦相似度的分子部分，
    # 分母则是用户间分别交互的物品数量乘积
    for u, users in tqdm(sim.items()):
        for v, score in users.items():
            sim[u][v] = score / sqrt(mp[u] * mp[v])

    # 4.通过测试集为每个用户推荐相似度最高的前 N 个物品
    # 首先需要通过用户相似度矩阵得到与当前用户最相似的 K 个用户
    # 根据这 K 个用户存在交互的物品，并从其中排除与测试集用户存在交互的物品，计算相似度分数
    # 计算结果是多个用户对同一物品的累加和
    items_rank = {}
    for u, _ in tqdm(test_user_items.items()):  # 遍历测试集的每个用户
        items_rank[u] = {}
        for v, score in sorted(sim[u].items(), key=lambda x: x[1], reverse=True)[:K]:  # 从用户相似度矩阵中选取与当前用户最相似的前 K 个用户
            for item in train_user_items[v]:  # 遍历这 K 个用户
                if item not in train_user_items[u]:  # 如果用户相似度矩阵涉及的用户在测试集涉及用户出现过，则不予推荐
                    if item not in items_rank[u]:
                        items_rank[u][item] = 0
                    items_rank[u][item] += score  # 累加这 K 个用户对同一个物品的评分

    items_rank_tmp = {}
    for k, v in items_rank.items():
        sorted_items = sorted(v.items(), key=lambda x: x[1], reverse=True)[:N]
        items_set = set([x[0] for x in sorted_items])
        items_rank_tmp[k] = items_set
    items_rank = items_rank_tmp

    return items_rank


if __name__ == '__main__':
    root_path = '../data/ml-1m/'
    train_user_items, test_user_items = load_data(root_path)
    rec_items = User_CF_Rec(train_user_items, test_user_items, 80, 10)
    RecEval.rec_eval(train_user_items, test_user_items, rec_items)
