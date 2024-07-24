from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def load_data(path):
    # 读取数据
    data = pd.read_csv(path)
    # 获取特征
    columns = data.columns.values
    dense_features = [feature for feature in columns if 'I' in feature]
    sparse_features = [feature for feature in columns if 'C' in feature]
    return data, sparse_features, dense_features


def data_preprocess(data, sparse_features, dense_features):
    data[dense_features] = data[dense_features].fillna(0.0)
    for f in dense_features:
        data[f] = data[f].apply(lambda x: np.log(x + 1.0) if x > -1.0 else -1.0)

    data[sparse_features] = data[sparse_features].fillna("-1")
    lbe = LabelEncoder()
    for f in dense_features:
        data[f] = lbe.fit_transform(data[f])

    return data[dense_features + sparse_features]


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dim,), name=fc.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(feature_columns, sparse_input, is_linear):
    # 定义一个 embedding 层对应的字典
    embedding_layers_dict = dict()
    # 筛选所有特征中的 sparse 特征
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 如果是用于线性部分的 embedding 层，其维度为 1，否则维度就是自己定义的 embedding 属性
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)
    return embedding_layers_dict


def get_linear_logits(dense_input, sparse_input, linear_sparse_feature_columns):
    # 将 linear 部分所有 dense 特征生成的 Input 层，通过全连接层后得到 dense 特征的线性对数 linear logits
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input.values()))
    dense_logits_output = Dense(1)(concat_dense_inputs)

    # 将 linear 部分所有 sparse 特征生成 embedding 层
    # 原因：如果在 linear 部分对特征直接进行 one-hot 编码然后通过全连接层，一旦维度非常大，会使得计算非常慢；
    # 但如果对特征生成 embedding 层，就可以直接通过查表的方式获取到非零元素对应的权重，然后直接相加，效率非常高
    linear_embedding_layers = build_embedding_layers(linear_sparse_feature_columns, sparse_input, is_linear=True)

    # 拼接一维的 embedding 层
    # 注意使用 Flatten 层对应维度
    sparse_1d_embedding = []
    for fc in linear_sparse_feature_columns:
        feature_input = sparse_input[fc.name]
        embedding_layer = Flatten()(linear_embedding_layers[fc.name](feature_input))
        sparse_1d_embedding.append(embedding_layer)

    # 由于在 embedding 层查询得到的权重即为对应 one-hot 向量中一个位置的权重，所以就不用进行全连接了，本身一维的 embedding 层就相当于全连接了
    # 但此处的输入特征仅有 0 和 1 组成，所以直接累加非零元素的对应权重就等同于通过了全连接层
    sparse_logits_output = Add()(sparse_1d_embedding)

    # 将 dense 特征和 sparse 特征对应的 logits 相加，得到最终 linear 部分的 logits
    linear_logits = Add()([dense_logits_output, sparse_logits_output])
    return linear_logits


def concat_embedding_list(dnn_sparse_feature_columns, sparse_input, dnn_embedding_layers, flatten):
    # 筛选 sparse 特征
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_sparse_feature_columns))

    dnn_embedding_layers_list = []
    for fc in sparse_feature_columns:
        input = sparse_input[fc.name]  # 获取 input 层
        embedding_layer = dnn_embedding_layers[fc.name](input)  # 将 input 层输入到对应的 embedding 层中
        # 如果 embedding_list 最终是直接输入到 Dense 层，则需要进行 Flatten，否则不需要
        if flatten:
            embedding_layer = Flatten()(embedding_layer)
        dnn_embedding_layers_list.append(embedding_layer)

    return dnn_embedding_layers_list


def get_dnn_logits(dense_input, sparse_input, dnn_sparse_feature_columns, dnn_embedding_layers):
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input.values()))

    sparse_kd_embedding_list = concat_embedding_list(dnn_sparse_feature_columns, sparse_input, dnn_embedding_layers,
                                                     flatten=True)
    concat_sparse_kd_embedding = Concatenate(axis=1)(sparse_kd_embedding_list)
    # dnn 层
    dnn_input = Concatenate(axis=1)([concat_dense_inputs, concat_sparse_kd_embedding])
    # dnn 层 的 Dropout 参数，Dense 中的参数及 Dense 的层数可自行设定
    dnn_out = Dropout(0.5)(Dense(1024, activation='relu')(dnn_input))
    dnn_out = Dropout(0.3)(Dense(512, activation='relu')(dnn_out))
    dnn_out = Dropout(0.1)(Dense(256, activation='relu')(dnn_out))

    dnn_logits = Dense(1)(dnn_out)
    return dnn_logits


def Wide_Deep(linear_feature_columns, dnn_feature_columns):
    # 接收特征并构建模型的输入层
    dense_input, sparse_input = build_input_layers(linear_feature_columns + dnn_feature_columns)
    input_layers = list(dense_input.values()) + list(sparse_input.values())

    # 将特征输入到 Wide 部分
    # 筛选 linear 部分的 sparse 特征，用于做 1 维的 embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))
    linear_logits = get_linear_logits(dense_input, sparse_input, linear_sparse_feature_columns)

    # 构建维度为 k 的 embedding 层
    dnn_embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input, is_linear=False)

    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # 将特征输入到 Deep 部分，这一部分的输入是将 dense 特征和 embedding 特征拼在一起输入到 dnn 中
    dnn_logits = get_dnn_logits(dense_input, sparse_input, dnn_sparse_feature_columns, dnn_embedding_layers)

    # 将linear,dnn的logits相加作为最终的logits
    output_logits = Add()([linear_logits, dnn_logits])

    # 这里的激活函数使用sigmoid
    output_layers = Activation("sigmoid")(output_logits)

    model = Model(inputs=input_layers, outputs=output_layers)
    return model


if __name__ == '__main__':
    path = '../data/criteo_sample.txt'
    data, sparse_features, dense_features = load_data(path)
    train_data = data_preprocess(data, sparse_features, dense_features)
    train_data = train_data.insert(0, 'label', data['label'])

    # 对特征进行分组，并对分组后的特征做标记
    # 可根据实际场景选择
    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'dim'])
    # linear 部分
    linear_feature_columns = [SparseFeat(name=feature, vocabulary_size=data[feature].nunique(), embedding_dim=4) for
                              i, feature in enumerate(sparse_features)] + [DenseFeat(name=feature, dim=1) for
                                                                           feature in dense_features]
    # dnn 部分
    dnn_feature_columns = [SparseFeat(feature, vocabulary_size=data[feature].nunique(), embedding_dim=4) for i, feature
                           in enumerate(sparse_features)] + [DenseFeat(feature, 1) for feature in dense_features]
    model = Wide_Deep(linear_feature_columns, dnn_feature_columns)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}

    # 训练模型
    model.fit(train_model_input, train_data['label'].values, batch_size=64, epochs=5, validation_split=0.2)
