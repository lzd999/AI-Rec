# Wide & Deep

## 动机

考虑这样一个场景：用户与物品交互产生的历史数据，对数据预处理后生成的矩阵非常稀疏且维度很高。

如果使用 LR 模型，优点是能记住每个特征的 CTR，缺点是需要人为进行特征工程，泛化能力差，无法学习到训练集未出现过的特征的交叉信息；如果使用
FM 模型，计算效率高，可以学习到训练集未出现过的特征的交叉信息，但是在这种场景下很难学习到有效的 Embedding 表示，导致模型过度泛化；因此提出了
Wide & Deep 模型。

2016年，Google 提出了 Wide & Deep 模型，顾名思义，该模型结合了 Wide 模块和 Deep 模块。

Wide 模块主要负责**接收经过编码后的类别型特征**，类别型特征能很好地标识某一样本，更清晰地表达样本信息。

Deep 模块主要负责**接收经过特征工程后的数值型特征**，通过神经网络端对端的拟合特性使得数值型特征能进行充分融合，提高对样本信息的表达效果。

结合后产生的 Wide & Deep 模型很好地吸收这两个模块对类别型特征和数值型特征在表达样本信息方面的优点。

## 记忆能力 & 泛化能力

记忆能力（$Memorization$）和泛化能力（$Generalization$）是推荐系统领域比较常见的两个概念，其中：

记忆能力是指**模型通过用户与物品的交互行为生成的矩阵学习到高频共现的特征组合的能力**；协同过滤模型和 MF
模型就是典型的记忆能力比较强的模型，通过学习用户或物品间的交互行为生成推荐结果。

泛化能力是指**模型利用用户或者物品特征之间的传递性去探索历史数据中从未出现过的特征组合的能力**；FM
模型则是典型的泛化能力比较强的模型，通过特征向量的一阶和二阶交互生成推荐结果。

而今天介绍的 Wide & Deep 模型则是兼具记忆能力和泛化能力的模型。

## 原理

<div align=center>
<img src="http://ryluo.oss-cn-chengdu.aliyuncs.com/Javaimage-20200910214310877.png" alt="image-20200910214310877" style="zoom:65%;" />
</div>

如果有一定的机器学习和深度学习基础，很容易看懂 Wide & Deep 模型的结构。难点在于如何根据业务场景需求，将哪些特征放在 Wide
模块，哪些特征放在 Deep 模块，就因人而异了，这也是该模型推荐效果高低的前提。

**如何理解 Wide 模块能增强模型的“记忆能力”，Deep模块能增强模型的“泛化能力”？**

1.Wide 模块是一个广义的**线性模型**，其输入的特征主要由两部分组成，一部分是原始特征的部分特征，另一部分是原始特征的交叉特征，交叉特征可以定义为：
$$
\phi_{k}(x)=\prod_{i=1}^d x_i^{c_{ki}}, c_{ki}\in \{0,1\}
$$

其中 $c_{ki}$ 是一个布尔变量，表示当第 $i$ 个特征属于第 $k$ 个特征组合时，$c_{ki}$ 的值为 1，否则为 0；$x_i$ 是第 $i$
个特征的值，表示当且仅当两个特征同时为 1，$x_i$ 的值为 1，其它情况都为 0，本质上就是一个特征组合。

Wide 模块训练时使用的优化器是带 $L_1$ 正则的 $FTRL$ （$Follow\space The \space Regularized\space Leader$）算法，而
$FTRL\space with\space L_1$ 算法是非常注重模型的稀疏性质的，相当于 Wide & Deep 模型采用该算法就是为了让 Wide
模块变得更加稀疏，这就大大压缩了模型权重和特征向量的维度。

因此 **Wide 模块训练完成后得到的特征非常重要**，**模型的“记忆能力”非常高**，具备很好的可解释性：发现“直接的”、“暴力的”、“显然的”关联规则的能力。例如
Google Wide & Deep 模型期望 Wide 模块发现如下规则：用户安装了应用 A，如果此时向用户推荐应用 B，则用户安装应用 B 的概率非常大。

2.Deep 模块是一个 DNN 模型，输入的特征主要分为两大类，一类是数值特征（可直接输入DNN），一类是类别特征（需要经过 Embedding
后才能输入到 DNN），Deep 模块的数学形式如下：
$$
a^{(l+1)} = f(W^{l}a^{(l)} + b^{l})
$$
**由于 DNN 模型随着训练层数的增加，每层训练完成后得到的中间特征就越抽象，这就提高了模型的泛化能力。**Deep 模块使用的 DNN
模型，Google 为了使模型可以得到更精确的解，使用了深度学习常用的优化器 AdaGrad

**Wide部分与Deep部分的结合**

Wide&Deep 模型是将两部分输出的结果结合起来联合训练，将 deep 和 wide 部分的输出重新使用一个 LR 模型做最终的预测，输出概率值。联合训练的数学公式如下：
$$
P(Y=1|x)=\delta(w_{wide}^T[x,\phi(x)] + w_{deep}^T a^{(lf)} + b)
$$

需要注意的是，因为 Wide 侧的数据是高维稀疏的，所以模型作者使用了 FTRL 算法优化，而 Deep 侧使用了 Adagrad 优化。

## 实现

tensorflow 库内置了 Wide&Deep 模型，即

```python
tf.keras.experimental.WideDeepModel(
    linear_model, dnn_model, activation=None, **kwargs
)
```

很容易可以看出，Wide&Deep 模型就是将 linear_model 和 dnn_model 拼接在一起，对应了 W&D 的整体流程

```python
linear_model = LinearModel()
dnn_model = keras.Sequential([keras.layers.Dense(units=64),
                             keras.layers.Dense(units=1)])
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
# define dnn_inputs and linear_inputs as separate numpy arrays or
# a single numpy array if dnn_inputs is same as linear_inputs.
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
# or define a single `tf.data.Dataset` that contains a single tensor or
# separate tensors for dnn_inputs and linear_inputs.
dataset = tf.data.Dataset.from_tensors(([linear_inputs, dnn_inputs], y))
combined_model.fit(dataset, epochs)
```

从源码注释给出的例子看：

第一步是**调用了 keras.experimental 中的 LinearModel 类**；

第二步是**简单实现了一个全连接神经网络 DNN**；

第三步是**使用 WideDeepModel 将前两步产生的两个 model 拼接在一起**，然后进行常规的 compile 和 fit。

也可以在拼接前分别进行训练：

```python
linear_model = LinearModel()
linear_model.compile('adagrad', 'mse')
linear_model.fit(linear_inputs, y, epochs)
dnn_model = keras.Sequential([keras.layers.Dense(units=1)])
dnn_model.compile('rmsprop', 'mse')
dnn_model.fit(dnn_inputs, y, epochs)
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
```

可以看出，前三行代码训练了 linear_model，中间三行训练了 dnn_model，最后三行将前两个 model 拼接

