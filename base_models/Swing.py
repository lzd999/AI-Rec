from collections import defaultdict
from itertools import combinations

import pandas as pd


def load_data(root_path):
    column_names = ['user_id', 'movie_id', 'rating']
    train_data = pd.read_csv(root_path, sep=', ', engine='python', names=column_names)
    return train_data


def get_user_items_user(train_data):
    user_items = defaultdict(set)
    items_user = defaultdict(set)
    for i, row in train_data.iterrows():
        user_items[row['user_id']].add(row['movie_id'])
        items_user[row['movie_id']].add(row['user_id'])
    return user_items, items_user


def Swing_Rec(user_items, items_user, alpha, N):
    items_pairs = list(combinations(items_user.keys(), 2))  # 生成 (物品, 物品) 的全排列对
    items_dict = defaultdict(dict)
    for (i, j) in items_pairs:
        # 与物品 i 存在交互的用户和与物品 j 存在交互的用户取交集后生成 (用户, 用户) 的全排列对
        user_pairs = list(combinations(items_user[i] & items_user[j], 2))
        res = 0
        for (u, v) in user_pairs:
            res += 1 / (alpha + len(list(user_items[u] & user_items[v])))  # Swing 算法公式
        if res != 0:
            items_dict[i][j] = format(res, '.6f')


if __name__ == "__main__":
    train_data_path = "../data/record.txt"
    train_data = load_data(train_data_path)
    user_items, items_user = get_user_items_user(train_data)
    alpha = 1.0
    N = 10
    rec_items = Swing_Rec(user_items, items_user, alpha, N)
