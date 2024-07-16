import os

import faiss
import numpy as np
import pandas as pd
from Rec.base_models.MetricsEvaluation import RecEval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def load_data(path):
    # 从文件导入数据
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(path, 'ratings.dat'), sep='::', engine='python', names=column_names)
    # 对数据集使用 one-hot 编码，便于后续模型训练
    lbe = LabelEncoder()
    data['user_id'] = lbe.fit_transform(data['user_id'])
    data['movie_id'] = lbe.fit_transform(data['movie_id'])
    # 划分训练集和验证集
    train_data, validate_data = train_test_split(data, test_size=0.2)
    train_grouped = train_data.groupby('user_id').agg({
        'movie_id': lambda x: list(x),
        'rating': lambda x: list(x)
    }).reset_index()
    validate_grouped = validate_data.groupby('user_id').agg({
        'movie_id': lambda x: list(x),
        'rating': lambda x: list(x)
    }).reset_index()
    # 生成训练集每个用户对哪些电影有过评分
    train_user_items = dict()
    for user, items in zip(*(list(train_grouped['user_id']), list(train_grouped['movie_id']))):
        train_user_items[user] = set(items)
    # 生成验证集每个用户对哪些电影有过评分
    validate_user_items = dict()
    for user, items in zip(*(list(validate_grouped['user_id']), list(validate_grouped['movie_id']))):
        validate_user_items[user] = set(items)
    return train_data, validate_data, train_user_items, validate_user_items


def MF(nums_user, nums_item, embedding_dim):
    # 清除当前 Keras 会话中包括模型、权重和优化器的所有状态，防止多次构建和训练模型时出现内存泄露
    clear_session()
    # 定义输入层，表示用户和物品
    users_input = Input(shape=[None, ])
    items_input = Input(shape=[None, ])
    # 定义嵌入层，将用户和物品的稀疏ID映射到对应的低维向量中
    users_embedding = Embedding(nums_user, embedding_dim)(users_input)
    items_embedding = Embedding(nums_item, embedding_dim)(items_input)
    # 对用户和物品向量批量归一化
    users = BatchNormalization()(users_embedding)
    items = BatchNormalization()(items_embedding)
    # 定义重塑层，将嵌入层的输出从二维张量（批次大小，嵌入维度）转化为一维张量（嵌入维度）
    users = Reshape((embedding_dim,))(users)
    items = Reshape((embedding_dim,))(items)
    # 定义点积层，预测用户-电影的交互评分
    output = Dot(1)([users, items])
    # 构建训练模型，指定输入和输出
    model = Model(inputs=[users_input, items_input], outputs=output)
    # 模型编译采用均方误差 mse 作为损失函数，adam 优化器进行优化
    model.compile(loss='mse', optimizer='adam')
    # 打印模型结构摘要，显示每一层的参数数量和连接方式
    # model.summary()
    # 为了方便获取模型中的某些层，进行如下属性设置
    model.__setattr__('users_input', users_input)
    model.__setattr__('items_input', items_input)
    model.__setattr__('users_embedding', users_embedding)
    model.__setattr__('items_embedding', items_embedding)

    return model


if __name__ == '__main__':
    # 导入数据
    root_path = '../data/ml-1m/'
    train_data, validate_data, train_user_items, validate_user_items = load_data(root_path)
    # 获取训练集 user 和 item 的数量
    nums_user = train_data['user_id'].max() + 1
    nums_item = train_data['movie_id'].max() + 1
    # 定义 user 和 item 的向量维度
    embedding_dim = 64
    # 设置矩阵分解模型相关参数
    model = MF(nums_user, nums_item, embedding_dim)
    # 定义矩阵分解模型路径
    model_path = 'MF_model.h5'
    # 定义模型训练期间的监控参数
    # 在模型训练期间，自动保存使得验证集损失函数值最小的模型权重
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )
    # 在模型训练期间，如果验证集损失函数值在连续 5 个训练轮数中都没有减少超过 0.0001，则提前停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=1,
        mode='min'
    )
    # 在模型训练期间，如果验证集损失函数值连续 3 个训练轮数都没有减少，则学习率减少 1/2
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.0001,
        cooldown=0
    )
    callbacks = [checkpoint, reduce_lr, early_stopping]
    # 训练矩阵分解模型
    model.fit([train_data['user_id'].values, train_data['movie_id'].values],
              train_data['rating'].values,
              batch_size=256,
              epochs=1,
              validation_split=0.1,
              callbacks=callbacks,
              verbose=1,
              shuffle=True)
    # 获取通过训练集训练模型时用户和电影矩阵的嵌入层
    users_embedding_model = Model(inputs=model.users_input, outputs=model.users_embedding)
    items_embedding_model = Model(inputs=model.items_input, outputs=model.items_embedding)
    # 提取训练集的唯一物品id，验证集的唯一用户id，均按升序排序
    # 方便后续按顺序遍历验证集的每个用户和训练集的每个物品，和推荐系统预测结果一一对应
    asc_train_items = sorted(train_data['movie_id'].unique())
    asc_validate_users = sorted(train_data['user_id'].unique())
    # 由于后续需要使用向量搜索库 Faiss，为了方便后续将 Faiss 库返回的相对索引正确映射回训练集的原始索引
    # 需要提前保存每个物品在训练集的原始索引
    raw_idx_train_items = dict()
    for i, item in enumerate(asc_train_items):
        raw_idx_train_items[i] = item
    # 训练集的唯一物品集合
    items_set = set(train_data['movie_id'].unique())
    # 通过 predict 获取验证集的用户向量和训练集的物品向量
    users_predict = users_embedding_model.predict([asc_validate_users], batch_size=256).squeeze(axis=1)
    items_predict = items_embedding_model.predict([asc_train_items], batch_size=256).squeeze(axis=1)
    # 使用向量搜索库进行最近邻搜索
    # IndexFlatIP 表示使用向量内积作为相似度量
    # ascontiguousarray 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.ascontiguousarray(items_predict))
    K = 80
    N = 10
    # 为每个用户嵌入向量找到最相似的 K 个物品嵌入向量，并返回距离数组 D 和对应索引 I
    D, I = index.search(np.ascontiguousarray(users_predict), K)
    # 将推荐结果转化成可评价的指标
    rec_items = dict()
    for i, u in enumerate(asc_validate_users):
        items = list(map(lambda x: raw_idx_train_items[x], list(I[i])))
        items = list(filter(lambda x: x not in train_user_items[u], items))[:N]
        rec_items[u] = set(items)
    # 评估指标控制台输出
    RecEval.rec_eval(train_data, validate_data, rec_items)
