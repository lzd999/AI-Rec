import gc

import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(path):
    # 读取数据
    train_df = pd.read_csv(path + "kaggle_train.csv")
    test_df = pd.read_csv(path + "kaggle_test.csv")
    # 对数据做预处理
    # 去掉id列，并把训练集和测试集合并，填充缺失值
    train_df.drop(labels=['Id'], axis=1, inplace=True)
    train_df.fillna(-1, inplace=True)
    test_df.drop(labels=['Id'], axis=1, inplace=True)
    test_df.insert(0, 'Label', -1)
    test_df.fillna(-1, inplace=True)
    data = pd.concat([train_df, test_df])
    # 将特征列分组处理
    continuous_feature = ['I' + str(i + 1) for i in range(13)]
    categorical_feature = ['C' + str(i + 1) for i in range(26)]
    return data, continuous_feature, categorical_feature


def LR(data, continuous_feature, categorical_feature):
    # 对连续型特征使用最大最小归一化
    scaler = MinMaxScaler()
    data[continuous_feature] = scaler.fit_transform(data[continuous_feature].values)
    # 对离散型特征使用 one-hot 编码
    # for feature in categorical_feature:
    #     one_hot_feature = pd.get_dummies(data[feature], prefix=feature)
    #     data.drop([feature], axis=1, inplace=True)
    #     data = pd.concat([data, one_hot_feature], axis=1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one_hot_feature = ohe.fit_transform(data[categorical_feature].values.astype(str))
    one_hot_cols = [f"{category}_{cat}" for category, cats in zip(categorical_feature, ohe.categories_) for cat in cats]
    one_hot_df = pd.DataFrame(one_hot_feature, columns=one_hot_cols, index=data.index)
    data = pd.concat([data.drop(columns=categorical_feature), one_hot_df], axis=1)

    train_data = data[data["Label"] != -1]
    test_data = data[data["Label"] == -1]
    train_label_col = train_data.pop('Label')
    test_data.drop(['Label'], axis=1, inplace=True)
    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_label_col, test_size=0.2, random_state=2020)
    # 建立 LR 模型并进行训练
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # 计算损失函数
    train_log_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])  # log_loss => −(ylog(p)+(1−y)log(1−p))
    val_log_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    # 使用 LR 模型预测
    y_pred = lr.predict_proba(test_data)[:, 1]

    print('LR:')
    print('train_log_loss: ', train_log_loss)
    print('val_log_loss: ', val_log_loss)
    print("y_pred: ", y_pred[:10])


def GBDT(data, continuous_feature, categorical_feature):
    # 对离散型特征使用 one-hot 编码
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one_hot_feature = ohe.fit_transform(data[categorical_feature].values.astype(str))
    one_hot_cols = [f"{category}_{cat}" for category, cats in zip(categorical_feature, ohe.categories_) for cat in cats]
    one_hot_df = pd.DataFrame(one_hot_feature, columns=one_hot_cols, index=data.index)
    data = pd.concat([data.drop(columns=categorical_feature), one_hot_df], axis=1)
    # 分离训练集和测试集
    train_data = data[data['Label'] != -1]
    train_label_col = train_data.pop('Label')
    test_data = data[data['Label'] == -1]
    test_data.drop(['Label'], axis=1, inplace=True)
    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_label_col, test_size=0.2, random_state=2020)
    # 建立 GBDT 模型
    gbdt = lgb.LGBMClassifier(num_leaves=100,
                              max_depth=12,
                              learning_rate=0.01,
                              n_estimators=1000,
                              objective='binary',
                              subsample=0.8,
                              min_child_weight=0.5,
                              colsample_bytree=0.7, )
    gbdt.fit(x_train, y_train,
             eval_set=[(x_train, y_train), (x_val, y_val)],
             eval_names=['train', 'val'],
             eval_metric='binary_log_loss')
    # 计算训练集和验证集的对数损失
    train_log_loss = log_loss(y_train, gbdt.predict_proba(x_train)[:, 1])
    val_log_loss = log_loss(y_val, gbdt.predict_proba(x_val)[:, 1])
    # 模型预测
    y_pred = gbdt.predict_proba(test_data)[:, 1]

    print('GBDT:')
    print('train_log_loss: ', train_log_loss)
    print('val_log_loss: ', val_log_loss)
    print('y_pred:', y_pred[:10])


def GBDT_LR(data, continuous_feature, categorical_feature):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one_hot_feature = ohe.fit_transform(data[categorical_feature].values.astype(str))
    one_hot_cols = [f"{category}_{cat}" for category, cats in zip(categorical_feature, ohe.categories_) for cat in cats]
    one_hot_df = pd.DataFrame(one_hot_feature, columns=one_hot_cols, index=data.index)
    data = pd.concat([data.drop(columns=categorical_feature), one_hot_df], axis=1)

    train_data = data[data['Label'] != -1]
    train_label_col = train_data.pop('Label')
    test_data = data[data['Label'] == -1]
    test_data.drop(['Label'], axis=1, inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(train_data, train_label_col, test_size=0.2, random_state=2020)

    gbdt = lgb.LGBMClassifier(num_leaves=100,
                              max_depth=12,
                              learning_rate=0.01,
                              n_estimators=1000,
                              objective='binary',
                              subsample=0.8,
                              min_child_weight=0.5,
                              colsample_bytree=0.7, )
    gbdt.fit(x_train, y_train,
             eval_set=[(x_train, y_train), (x_val, y_val)],
             eval_names=['train', 'val'],
             eval_metric='binary_log_loss')
    # model = gbdt.booster_

    # 使用 GBDT 模型预测训练集和测试集特征
    gbdt_train_features = gbdt.predict(train_data, pred_leaf=True)
    gbdt_test_features = gbdt.predict(test_data, pred_leaf=True)
    # 通过训练集获取所有特征名并生成 df
    gbdt_features_leaves = ['gbdt_leaf_' + str(i) for i in range(gbdt_train_features.shape[1])]
    gbdt_train_df = pd.DataFrame(gbdt_train_features, columns=gbdt_features_leaves)
    gbdt_test_df = pd.DataFrame(gbdt_test_features, columns=gbdt_features_leaves)

    train_data = pd.concat([train_data, gbdt_train_df], axis=1)
    test_data = pd.concat([test_data, gbdt_test_df], axis=1)
    train_len = train_data.shape[0]
    data = pd.concat([train_data, test_data])
    del train_data, test_data
    gc.collect()

    scaler = MinMaxScaler()
    data[continuous_feature] = scaler.fit_transform(data[continuous_feature].values)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    gbdt_features_leaves_data = ohe.fit_transform(data[gbdt_features_leaves])
    gbdt_features_leaves_df = pd.DataFrame(gbdt_features_leaves_data,
                                           columns=ohe.get_feature_names_out(input_features=gbdt_features_leaves))
    data = data.drop(gbdt_features_leaves, axis=1)
    data = data.merge(gbdt_features_leaves_df, left_index=True, right_index=True)

    train_data = data[:train_len]
    test_data = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train_data, train_label_col, test_size=0.2, random_state=2020)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    train_log_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    val_log_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    y_pred = lr.predict_proba(test_data)[:, 1]

    print('GBDT+LR:')
    print('train_log_loss: ', train_log_loss)
    print('val_log_loss: ', val_log_loss)
    print('y_pred:', y_pred[:10])


if __name__ == '__main__':
    path = '../data/'
    data, continuous_feature, categorical_feature = load_data(path)
    LR(data.copy(), continuous_feature, categorical_feature)
    GBDT(data.copy(), continuous_feature, categorical_feature)
    GBDT_LR(data.copy(), continuous_feature, categorical_feature)
