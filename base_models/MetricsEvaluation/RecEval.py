import math

"""
衡量 CF 好坏的评价指标
"""


def Recall(rec_dict, items_dict):
    """
    召回率
    :param rec_dict: 当前推荐算法生成的推荐物品列表
    :param items_dict: 用户实际点击的物品列表
    """
    hit_items = 0
    all_items = 0
    for uid, items in items_dict.items():
        rel_set = items
        rec_set = rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rel_set)
    return round(hit_items / all_items * 100, 2)


def Precision(rec_dict, items_dict):
    """
    精确率
    :param rec_dict: 当前推荐算法生成的推荐物品列表
    :param items_dict: 用户实际点击的物品列表
    """
    hit_items = 0
    all_items = 0
    for uid, items in items_dict.items():
        rel_set = items
        rec_set = rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rec_set)
    return round(hit_items / all_items * 100, 2)


def Coverage(rec_dict, train_dict):
    """
    覆盖率
    :param rec_dict: 当前推荐算法生成的推荐物品列表
    :param train_dict: 训练集用户实际点击的物品列表
    """
    rec_items = set()
    all_items = set()
    for uid in rec_dict:
        for item in train_dict[uid]:
            all_items.add(item)
        for item in rec_dict[uid]:
            rec_items.add(item)
    return round(len(rec_items) / len(all_items) * 100, 2)


def Popularity(rec_dict, train_dict):
    """
    新颖度
    :param rec_dict: 当前推荐算法生成的推荐物品列表
    :param train_dict: 训练集用户实际点击的物品列表
    """
    pop_items = {}
    for uid in train_dict:
        for item in train_dict[uid]:
            if item not in pop_items:
                pop_items[item] = 0
            pop_items[item] += 1
    pop = nums = 0
    for uid in rec_dict:
        for item in rec_dict[uid]:
            pop += math.log(pop_items[item] + 1)  # 物品流行度满足长尾分布，取对数使得均值更稳定
            nums += 1
    return round(pop / nums * 100, 2)


def rec_eval(train_user_items, test_user_items, rec_items):
    print("recall:", Recall(rec_items, test_user_items))
    print("precision:", Precision(rec_items, test_user_items))
    print("coverage:", Coverage(rec_items, train_user_items))
    print("popularity:", Popularity(rec_items, train_user_items))
