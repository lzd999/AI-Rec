import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tqdm import tqdm


class CrossLayer(Layer):
    def __init__(self, input_dim, output_dim=10, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        # 定义输入维度
        self.input_dim = input_dim
        # 定义输出维度
        self.output_dim = output_dim
        # 定义交叉特征的权重
        self.kernel = self.add_weight(name='kernel', shape=(input_dim, output_dim), initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return 0.5 * K.mean(a - b, axis=1, keepdims=True)


def process_feature(data, dense_features, sparse_features):
    df = data.copy()
    # 对密集特征取对数
    dense_df = df[dense_features].fillna(0.0)
    for f in tqdm(dense_features):
        dense_df[f] = dense_df[f].apply(lambda x: np.log(1 + x) if x > -1 else -1)
    dense_list = [dense_df]
    # 对稀疏特征进行 one_hot 编码
    sparse_df = df[sparse_features].fillna('-1')
    for f in tqdm(sparse_features):
        lbe = LabelEncoder()
        sparse_df[f] = lbe.fit_transform(sparse_df[f])
    sparse_list = []
    for f in tqdm(sparse_features):
        new_data = pd.get_dummies(sparse_df.loc[:, f].values)
        new_data.columns = [f + "_{}".format(i) for i in range(new_data.shape[1])]
        sparse_list.append(new_data)
    # 拼接预处理后的密集特征列和稀疏特征列
    new_df = pd.concat(dense_list + sparse_list, axis=1)
    return new_df


def load_data(path):
    # 读取数据
    data = pd.read_csv(path)
    # 过滤特征列的密集特征和稀疏特征
    cols = data.columns.values
    dense_feature_list = [f for f in cols if f[0] == 'I']
    sparse_feature_list = [f for f in cols if f[0] == 'C']
    # 对两种特征列做预处理
    features_df = process_feature(data, dense_feature_list, sparse_feature_list)
    return data, features_df


def FM(features_dim):
    # 定义输入层
    features_input = Input(shape=(features_dim,))
    # 定义全连接层用于处理一阶特征
    # units=1 表示输出维度为 1
    features_dense = (Dense(
        units=1,
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l2(0.01))
                      (features_input))
    # 自定义特征组合层用于处理二阶特征
    features_cross = CrossLayer(features_dim)(features_input)
    # 合并一阶特征和二阶特征
    features_sum = Add()([features_cross, features_dense])
    # 定义输出层
    features_output = Dense(units=1, activation='sigmoid')(features_sum)
    # 构建模型
    model = Model(inputs=features_input, outputs=features_output)
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    # model.summary()
    return model


if __name__ == "__main__":
    # 读取数据
    root_path = "../data/kaggle_train.csv"
    data, features_df = load_data(root_path)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(features_df, data['Label'], test_size=0.2, random_state=2020)
    # 定义 FM 模型
    model = FM(features_df.shape[1])
    # 开始训练模型
    model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=1, validation_data=(x_test, y_test))
    # 通过模型预测结果
    model.predict(x_test, batch_size=256, verbose=1)
    # 评估当前模型
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=256, verbose=1)
    # 保存模型
    model.save('FM_model.h5')
