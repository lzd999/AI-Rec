# 因子分解机 FM

## 概述

2010 年，Steffen Rendle 提出了 **FM**（Factorization Machine，因子分解机 / 因式分解机），其目的是为了解决传统的因子分解模型的部分缺点：

1.传统的因子分解模型每遇到一种新问题，都需要在矩阵分解 MF
的基础上建立新模型，从而推导出新的参数学习方法，并在学习过程中调整各种参数。然而可能会存在某个新问题，使得其对应的模型的参数数量繁多，类型复杂，对大多数编程人员来说这是费事，耗力且易错的。

2.传统的因子分解模型无法很好地利用特征工程来完成学习任务。

FM 则解决了上述缺点，它能够**通过特征向量去模拟因子分解模型**，不仅**结合了特征工程的普遍性和适用性**，还能够**利用因子分解模型对不同类别的变量间的交互作用进行建模**。

## 原理

1.LR 模型的缺点

LR 是对通过特征工程得到的特征做线性组合，然后传入 $Sigmoid$ 函数计算概率值，其本质是一种线性模型。然而大多数应用场景的特征不符合线性，LR
模型使用的每个特征都与最终输出结果独立，且需要人为计算特征的交叉相乘，非常麻烦。

2.FM 模型的改进

LR 模型的目标函数如下：
$$
y = \omega _{0}\space + \sum_{i=1}^{n} \omega _{i}x_{i}
$$
由于做手动交叉比较麻烦，因此考虑所有的二阶交叉项，即将目标函数修改为：
$$
y = \omega _{0} + \sum_{i=1}^{n} \omega _{i}x_{i}\space + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \omega _{ij}x_{i}x_{j}
$$
然而这个目标函数存在一个问题，即仅当 $x_{i}$ 和 $x_{j}$ 均不为 0 时二阶交叉项才会生效，FM 模型改进了这一问题

FM 模型使用了如下目标函数：
$$
y = \omega _{0} + \sum_{i=1}^{n} \omega _{i}x_{i}\space + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \left \langle v_{i},v_{j}
\right \rangle x_{i}x_{j}
$$
其中：

① $i$ 表示特征索引

② $n$ 表示特征数量

③ $x_i \in \mathbb{R}$ 表示第 $i$ 个特征的值

④ $v_i,v_j \in \mathbb{R}^{k} $ 分别表示特征 $x_i,x_j$ 对应的隐语义向量（Embedding向量）， $\left\langle v_{i}, v_
{j}\right\rangle=\sum_{f=1}^{k} v_{i, f} \cdot v_{j, f}$

⑤ $w_0,w_i\in \mathbb{R}$ 表示需要学习的参数

将 LR 模型和 FM 模型的目标函数对比，其实 FM 模型做出的改进是**把 $\omega_{ij}$ 替换成了 $\left\langle v_{i},v_{j}
\right\rangle $** ，**实质上是先为每一个 $x_{i}$ 计算一个 Embedding，再将两个向量之间的 Embedding 做内积**，**这种改进带来的好处是增强了模型的泛化能力**，使得即使某两个特征之前从未在训练集中同时出现，也不至于无法训练 $\omega_
{i}$！！！

## 算法

在 FM 模型的目标函数中，前两项为特征的一阶交互项，可以将其拆分为用户特征和物品特征的一阶特征交互项：
$$
\begin{aligned}
& \omega_{0}+\sum_{i=1}^{n} \omega_{i} x_{i} \\
&= \omega_{0} + \sum_{t \in I}\omega_{t} x_{t} + \sum_{u\in U}\omega_{u} x_{u} \\
\end{aligned}
$$
其中，$U$ 表示用户相关特征集合，$I$ 表示物品相关特征集合。

再观察第三项，为特征的二阶交互项，易得其计算复杂度为 $O\left(k n^{2}\right)$，为了降低计算复杂度，可进行如下变换：
$$
\begin{aligned}
& \sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle\mathbf{v}_{i}, \mathbf{v}_{j}\right\rangle x_{i} x_{j} \\
=& \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n}\left\langle\mathbf{v}_{i}, \mathbf{v}_{j}\right\rangle x_{i} x_
{j}-\frac{1}{2} \sum_{i=1}^{n}\left\langle\mathbf{v}_{i}, \mathbf{v}_{i}\right\rangle x_{i} x_{i} \\
=& \frac{1}{2}\left(\sum_{i=1}^{n} \sum_{j=1}^{n} \sum_{f=1}^{k} v_{i, f} v_{j, f} x_{i} x_{j}-\sum_{i=1}^{n} \sum_
{f=1}^{k} v_{i, f} v_{i, f} x_{i} x_{i}\right) \\
=& \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)^{}\left(\sum_{j=1}^{n} v_{j, f} x_
{j}\right)-\sum_{i=1}^{n} v_{i, f}^{2} x_{i}^{2}\right) \\
=& \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)^{2}-\sum_{i=1}^{n} v_{i, f}^{2} x_
{i}^{2}\right)
\end{aligned}
$$
对特征的二阶交互项进行变换后，带来的好处有：

1.FM 模型的二次项参数数量减少为 $kn$ 个，远少于变换前的 $kn^{2}$ 个，显著减少二次项参数数量

2.参数因子化使得 $x_{i}$ 和 $x_{j}$ 的参数不再相互独立，使得即使样本稀疏，FM 模型也能相对合理地估计二次项参数。

由于 FM 模型用于召回，因此可继续变换：
$$
\begin{aligned}
& \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)^{2}-\sum_{i=1}^{n} v_{i, f}^{2} x_
{i}^{2}\right) \\
=& \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{u \in U} v_{u, f} x_{u} + \sum_{t \in I} v_{t, f} x_{t}\right)^{2}-\sum_
{u \in U} v_{u, f}^{2} x_{u}^{2} - \sum_{t\in I} v_{t, f}^{2} x_{t}^{2}\right) \\
=& \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{u \in U} v_{u, f} x_{u}\right)^{2} + \left(\sum_{t \in I} v_{t, f} x_
{t}\right)^{2} + 2{\sum_{u \in U} v_{u, f} x_{u}}{\sum_{t \in I} v_{t, f} x_{t}} - \sum_{u \in U} v_{u, f}^{2} x_
{u}^{2} - \sum_{t \in I} v_{t, f}^{2} x_{t}^{2}\right)  
\end{aligned}
$$
其中，$U$ 表示用户相关特征集合，$I$ 表示物品相关特征集合。

将 FM 模型目标函数关于特征的一阶交互项和二阶交互项与用户和物品相关联后可以得到如下公式：
$$
y = \omega_{0} + \sum_{t \in I}\omega_{t} x_{t} + \sum_{u\in U}\omega_{u} x_{u} + \frac{1}{2} \sum_{f=1}^{k}\left(\left(
\sum_{u \in U} v_{u, f} x_{u}\right)^{2} + \left(\sum_{t \in I} v_{t, f} x_{t}\right)^{2} + 2{\sum_{u \in U} v_{u, f} x_
{u}}{\sum_{t \in I} v_{t, f} x_{t}} - \sum_{u \in U} v_{u, f}^{2} x_{u}^{2} - \sum_{t \in I} v_{t, f}^{2} x_
{t}^{2}\right)
$$
可以考虑将变换后计算得到的 $y$ 记作匹配分用于衡量用户和物品之间的匹配程度的高低。

在比较时，不同用户特征内部的一阶与二阶交互项一定是相同的，因此可以先排除；所以需要重点关注：

（1）**物品内部之间**的特征交互得分。（2）**用户和物品之间**的特征交互得分。

将全局偏置和用户特征内部的一、二阶特征交互项丢弃后，可以得到如下公式：
$$
\text{MatchScore}_{FM} = \sum_{t \in I} w_{t} x_{t} + \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{t \in I} v_{t, f} x_
{t}\right)^{2} - \sum_{t \in I} v_{t, f}^{2} x_{t}^{2}\right)  + \sum_{f=1}^{k}\left( {\sum_{u \in U} v_{u, f} x_
{u}}{\sum_{t \in I} v_{t, f} x_{t}} \right)
$$
在基于向量的召回模型中，为了使用 ANN（近似最近邻算法） 或 Faiss
加速查找与用户兴趣度匹配的物品。基于向量的召回模型，一般最后都会得到用户和物品的特征向量表示，然后通过向量之间的内积或者余弦相似度表示用户对物品的兴趣程度。而
FM 属于基于向量的召回模型，因此可转化为：
$$
\text{MatchScore}_{FM} = V_{item} V_{user}^T
$$

+ 用户向量：
  $$
  V_{user} = [1; \quad {\sum_{u \in U} v_{u} x_{u}}]
  $$

    + 用户向量由两项表达式拼接得到。
    + 第一项为常数 $1$，第二项是将用户相关的特征向量进行 sum pooling 。

+ 物品向量：
  $$
  V_
  {item} = [\sum_{t \in I} w_{t} x_{t} + \frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{t \in I} v_{t, f} x_{t}\right)^{2} - \sum_{t \in I} v_{t, f}^{2} x_{t}^{2}\right); \quad
  {\sum_{t \in I} v_{t} x_{t}} ]
  $$

    + 第一项表示物品相关特征向量的一阶、二阶特征交互。
    + 第二项是将物品相关的特征向量进行 sum pooling 。

**※**  如果直接将 FM 模型学习到的 User 的 Embedding 和 Item 的 Embedding 做内积设置成推荐系统的召回层，不是不行，但效果不会特别好。因为
**用户喜欢的，未必一定是与自身最匹配的，也包括一些自身性质极佳的item（e.g.,热门 item）**，所以非常有必要考虑**"
所有Item特征一阶权重之和"和“所有Item特征隐向量两两点积之和”**。

## **优缺点**

优点：

1.由于使用向量内积作为交叉特征的权重，使得即使数据非常稀疏也能有效地训练出交叉特征的权重

2.计算效率非常高（$O(n)$）

3.FM 模型的训练和预测均只需要处理样本中的非零特征，这加快了模型训练和线上预测的速度

4.由于 FM 模型的计算效率高，并且在稀疏场景下可以自动挖掘长尾低频物料，使得 FM
模型在召回、粗排和精排三个阶段都可以使用。应用在不同阶段时，样本构造、拟合目标和线上服务都有所不同。

缺点：

只能显式做特征的二阶交叉，无法做特征更高阶的交叉。

## 应用

目前 FM 模型在工业界的应用大多数步骤如下：

- 离线训练生成 FM 模型（学习目标可以是CTR）
- 取出训练好的 FM 模型的 Embedding
- 将每个 User 和 Item 对应的 Embedding 做avg pooling（平均）形成该用户和该物品的最终 Embedding
- 将所有的 Embedding 向量放入 Faiss 等
- 线上通过 uid 向推荐系统发起请求，推荐系统从离线生成的 FM 模型取出对应的 User embedding 并进行检索召回

