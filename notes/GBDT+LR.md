# GBDT+LR

基于协同过滤的召回模型以及后续改进的矩阵分解 MF
的缺点非常明显，那就是仅利用了用户与物品间存在的交互行为，而忽视了用户或者物品的自身特征以及所关联的上下文等，这会使得生成的推荐结果非常片面。为了生成较为全面的推荐结果，FaceBook
在 2014 年提出了 GBDT+LR 模型，该模型利用 GBDT 模型自动进行特征筛选和组合，进而生成新的离散特征向量，再把这些特征向量当做
LR 模型的输入，进而产生最后的推荐结果，可以看出，该模型能够综合利用用户、物品和上下文等多种不同的特征，生成较为全面的推荐结果，在
CTR 点击率预估场景下使用较为广泛。

## 一、GBDT 模型

### 原理

1999年，Friedman 提出了 GBDT 模型，即**梯度提升决策树**，是传统机器学习算法里对真实分布拟合最好的几个算法之一，在深度学习概念还未诞生时，GBDT
应用非常广泛，其优点在于：1.拟合效果好。2.既可以用于分类也可以用于回归。3.可以自定义筛选特征。

GBDT 模型通过采用加法模型（即基函数的线性组合），以及不断减小训练过程产生的误差来达到将数据分类或者回归的算法， 其训练过程如下图所示：

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200908202508786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center" style="zoom:65%;" />    
</div>

**通过多轮次迭代，每轮迭代 GBDT 都会产生一个弱分类器， 每个分类器在上一轮分类器的残差基础上进行训练**。通常 GBDT
模型对弱分类器的要求是**足够简单**， 并且满足低方差和高偏差。这是因为训练的过程是通过降低偏差来不断提高最终分类器的精度。因此，每个分类回归树的深度不会很深。迭代完成后得到的总分类器是将
**每轮训练得到的弱分类器**通过**加权求和**得到的（也就是加法模型）。

GBDT 模型如何进行二分类？首先一定要明确的是，GBDT 模型每轮的训练是在上一轮训练的残差基础之上进行再训练的，这里的残差指的是当前模型的负梯度值，这就使得
GBDT 模型每轮训练使用的弱分类器得到的输出结果相减是有意义的，GBDT 模型无论是分类还是回归都是使用的 **CART**
（Classification And Regression Tree，分类和回归树），既然是回归树，如何能解决二分类问题呢？

GBDT 模型解决二分类问题，其本质和 GBDT 模型解决回归问题是相同的：通过不断构建决策树，使预测结果一步步的接近目标值。区别在于解决二分类问题和解决回归问题所用的损失函数是不同的，在回归问题中，GBDT
模型每轮训练使用的损失函数通常是 MSE（Mean Square Error，均方损失误差）；而在分类问题中，GBDT 模型和 LR 模型一样，使用的损失函数公式如下：
$$
L=\arg \min
\left[\sum_{i}^{n}-\left(y_{i} \log \left(p_{i}\right)+\left(1-y_{i}\right) \log \left(1-p_{i}\right)\right)\right]
$$
其中，$y_i$ 是第 $i$ 个样本的观测值，取值要么是 $0$ 要么是 $1$；$p_i$ 是第 $i$ 个样本的预测值， 取值范围是 $[0, 1]$ 上的某个概率。

由于 GBDT 模型拟合的残差是当前模型的负梯度，因此要对该公式求导，即 $\frac{dL}{dp_i}$，对某个特定的样本 $l$，即
$\frac{dl}{dp_i}$：
$$
\begin{aligned}
l &=-y_{i} \log \left(p_{i}\right)-\left(1-y_{i}\right) \log \left(1-p_{i}\right) \\
&=-y_{i} \log \left(p_{i}\right)-\log \left(1-p_{i}\right)+y_{i} \log \left(1-p_{i}\right) \\
&=-y_{i}\left(\log \left(\frac{p_{i}}{1-p_{i}}\right)\right)-\log \left(1-p_{i}\right)
\end{aligned}
$$
如果熟悉 LR 模型，可以发现，$\left(\log \left(\frac{p_{i}}{1-p_{i}}\right)\right)$ 相当于对概率比取对数，在 LR
模型中这个式子的形式相当于 $\theta X$，后面才能推出 $p_i=\frac{1}{1+e^-{\theta X}}$ 这个形式。如果令
$\eta_i=\frac{p_i}{1-p_i}$，那么 $p_i=\frac{\eta_i}{1+\eta_i}$，上述公式就变成了：
$$
\begin{aligned}
l &=-y_{i} \log \left(\eta_{i}\right)-\log \left(1-\frac{e^{\log \left(\eta_{i}\right)}}{1+e^{\log \left(\eta_{i}\right)
}}\right) \\
&=-y_{i} \log \left(\eta_{i}\right)-\log \left(\frac{1}{1+e^{\log \left(\eta_{i}\right)}}\right) \\
&=-y_{i} \log \left(\eta_{i}\right)+\log \left(1+e^{\log \left(\eta_{i}\right)}\right)
\end{aligned}
$$
此时对 $log(\eta_i)$ 求导， 可得：
$$
\frac{d l}{d \log (\eta_i)}=-y_{i}+\frac{e^{\log \left(\eta_{i}\right)}}{1+e^{\log \left(\eta_{i}\right)}}=-y_i+p_i
$$
因此 GBDT 模型每轮训练得到的残差值为 $y_i-p_i$。

可以发现，GBDT 模型应用于分类问题的思想和 LR 模型的思想存在异曲同工之妙：**LR 模型是使用一个线性模型去拟合 $P(y=1|x)$
这个事件的对数概率 $log\frac{p}{1-p}=\theta^Tx$**；GBDT 模型应用于分类问题则是**使用一系列的梯度提升树去拟合某个事件 $P$
的对数概率**，其中 $P(Y=1 \mid x)=\frac{1}{1+e^{-F_{M}(x)}}$。

### 算法

GBDT 模型应用于分类问题的步骤具体如下：

1.**初始化 GBDT**

类似于 GBDT 模型应用于回归问题，GBDT 模型应用于分类问题的初始状态只有一个叶子节点 $F_{0}(x)$，该节点为所有样本的初始预测值。
$$
F_{0}(x)=\arg \min _{\gamma} \sum_{i=1}^{n} L(y, \gamma)
$$
其中，$F$ 代表 GBDT 模型，$F_{0}$ 代表模型的初始状态，该公式表达的意思是找到一个 $\gamma$ ∈ $[1,n]
$，使得所有样本计算得到的损失函数值最小，$\gamma$ 表示节点的输出，即叶子节点，且它是一个 $log(\eta_i)$
形式的值（回归值），初始状态时，$\gamma =F_0$。

这么解释可能还是有些抽象，看个例子。

假设当前有如下 3 个样本：

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200910095539432.png#pic_center" alt="在这里插入图片描述" style="zoom:80%;" /> 
</div>

现在想构建一棵 GBDT 分类树，它能通过「喜欢爆米花」、「年龄」和「颜色偏好」这 3 个特征来预测某一个样本是否喜欢看电影，应该怎么做？

首先，把数据代入公式中求损失函数值：
$$
\operatorname{Loss}=L(1, \gamma)+L(1, \gamma)+L(0, \gamma)
$$
对 $Loss$ 求导并令其等于 $0$，可以得到：
$$
\operatorname{Loss}=p-1 + p-1+p=0
$$
解得 $$p=\frac{2}{3}=0.67, \gamma=log(\frac{p}{1-p})=0.69$$，可算得模型的初始状态 $F_0(x)=0.69$。

2.**循环生成决策树**

回顾一下回归树的生成步骤：1.计算负梯度值得到残差；2.使用回归树拟合残差；3.计算叶子节点的输出值；4.更新模型。

1.计算负梯度得到残差
$$
r_{i m}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}\right]_{F(x)=F_
{m-1}(x)}
$$
此处使用 $m-1$ 棵树的模型，计算每个样本的残差 $r_{im}$, 也就是上面的 $y_i-pi$，对应到例子中就是：

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200910101154282.png#pic_center" alt="在这里插入图片描述" style="zoom:80%;" />
</div> 

2.使用回归树来拟合残差 $r_{im}$

此处 $i$ 表示样本，遍历每个特征，在每个特征下遍历每个取值，计算分裂后两组数据的均方损失误差， 找到最小的那个划分节点，对应到例子中就是：

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200910101558282.png#pic_center" alt="在这里插入图片描述" style="zoom:80%;" />
</div>
3.对每个叶子节点 $j$, 计算最佳的残差拟合值

公式如下：
$$
\gamma_{j m}=\arg \min _{\gamma} \sum_{x \in R_{i j}} L\left(y_{i}, F_{m-1}\left(x_{i}\right)+\gamma\right)
$$
表示在 2. 构建的回归树 $m$ 中，找到每个节点 $j$ 的输出 $\gamma_{j_m}$, 能使得该节点的 $loss$ 最小，具体来说，首先写出损失函数公式，以样本
1 为例，有：
$$
L\left(y_{1}, F_{m-1}\left(x_{1}\right)+\gamma\right)=-y_{1}\left(F_{m-1}\left(x_{1}\right)+\gamma\right)+\log \left(
1+e^{F_{m-1}\left(x_{1}\right)+\gamma}\right)
$$
这个式子就是上面推导的 $l$，因为我们要用回归树做分类，所以需要把分类的预测概率转换成对数概率回归的形式， 即 $log(\eta_i)
$，这就是模型的回归输出值。如果要求这个损失函数的最小值， 就需要先求导，再令导数等于 0 解得令损失函数值最小的 $\gamma$。

然而如果直接求导会非常麻烦，因此可以考虑使用**二阶泰勒公式近似表示该公式，再求导**，即将 $L(y_1, F_{m-1}(x_1))$当做常量 $f(
x)$， $\gamma$ 作为变量 $\Delta x$， 将 $f(x)$ 二阶展开，如下所示：
$$
f(x+\Delta x) \approx f(x)+\Delta x f^{\prime}(x)+\frac{1}{2} \Delta x^{2} f^{\prime \prime}(x)+O(\Delta x)
$$

$$
L\left(y_{1}, F_{m-1}\left(x_{1}\right)+\gamma\right) \approx L\left(y_{1}, F_{m-1}\left(x_{1}\right)\right)
+L^{\prime}\left(y_{1}, F_{m-1}\left(x_{1}\right)\right) \gamma+\frac{1}{2} L^{\prime \prime}\left(y_{1}, F_{m-1}\left(
x_{1}\right)\right) \gamma^{2} +O(\gamma)
$$

这时候再求导：
$$
\frac{d L}{d \gamma}=L^{\prime}\left(y_{1}, F_{m-1}\left(x_{1}\right)\right)+L^{\prime \prime}\left(y_{1}, F_{m-1}\left(
x_{1}\right)\right) \gamma
$$
要使 $Loss$ 最小，则令导数等于 0，可得：
$$
\gamma_{11}=\frac{-L^{\prime}\left(y_{1}, F_{m-1}\left(x_{1}\right)\right)}{L^{\prime \prime}\left(y_{1}, F_{m-1}\left(
x_{1}\right)\right)}
$$
可以看出，**分子即为残差**，**分母可通过对残差求导，得到原损失函数的二阶导**：
$$
\begin{aligned}
L^{\prime \prime}\left(y_{1}, F(x)\right) &=\frac{d L^{\prime}}{d \log (\eta_1)} \\
&=\frac{d}{d \log (\eta_1)}\left[-y_{i}+\frac{e^{\log (\eta_1)}}{1+e^{\log (\eta_1)}}\right] \\
&=\frac{d}{d \log (\eta_1)}\left[e^{\log (\eta_1)}\left(1+e^{\log (\eta_1)}\right)^{-1}\right] \\
&=e^{\log (\eta_1)}\left(1+e^{\log (\eta_1)}\right)^{-1}-e^{2 \log (\eta_1)}\left(1+e^{\log (\eta_1)}\right)^{-2} \\
&=\frac{e^{\log (\eta_1)}}{\left(1+e^{\log (\eta_1)}\right)^{2}} \\
&=\frac{\eta_1}{(1+\eta_1)}\frac{1}{(1+\eta_1)} \\
&=p_1(1-p_1)
\end{aligned}
$$
现在可以算出该节点的输出：
$$
\gamma_{11}=\frac{r_{11}}{p_{10}\left(1-p_{10}\right)}=\frac{0.33}{0.67 \times 0.33}=1.49
$$
其中 $\gamma_{jm}$ 表示第 $m$ 棵树的第 $j$ 个叶子节点。

接着是右边节点的输出，也就是样本 2 和样本 3 的输出，同样使用二阶泰勒公式展开：
$$
\begin{array}{l}
L\left(y_{2}, F_{m-1}\left(x_{2}\right)+\gamma\right)+L\left(y_{3}, F_{m-1}\left(x_{3}\right)+\gamma\right) \\
\approx L\left(y_{2}, F_{m-1}\left(x_{2}\right)\right)+L^{\prime}\left(y_{2}, F_{m-1}\left(x_{2}\right)\right)
\gamma+\frac{1}{2} L^{\prime \prime}\left(y_{2}, F_{m-1}\left(x_{2}\right)\right) \gamma^{2} \\
+L\left(y_{3}, F_{m-1}\left(x_{3}\right)\right)+L^{\prime}\left(y_{3}, F_{m-1}\left(x_{3}\right)\right)
\gamma+\frac{1}{2} L^{\prime \prime}\left(y_{3}, F_{m-1}\left(x_{3}\right)\right) \gamma^{2}
\end{array}
$$
求导， 再令导数为 0，可得到第 1 棵树的第 2 个叶子节点的输出：
$$
\begin{aligned}
\gamma_{21} &=\frac{-L^{\prime}\left(y_{2}, F_{m-1}\left(x_{2}\right)\right)-L^{\prime}\left(y_{3}, F_{m-1}\left(x_
{3}\right)\right)}{L^{\prime \prime}\left(y_{2}, F_{m-1}\left(x_{2}\right)\right)+L^{\prime \prime}\left(y_{3}, F_
{m-1}\left(x_{3}\right)\right)} \\
&=\frac{r_{21}+r_{31}}{p_{20}\left(1-p_{20}\right)+p_{30}\left(1-p_{30}\right)} \\
&=\frac{0.33-0.67}{0.67 \times 0.33+0.67 \times 0.33} \\
&=-0.77
\end{aligned}
$$
综合 $\gamma_{11}$ 和 $\gamma_{21}$，不难发现，我们可以直接计算每个叶子节点的输出：
$$
\gamma_{j m}=\frac{\sum_{i=1}^{R_{i j}} r_{i m}}{\sum_{i=1}^{R_{i j}} p_{i, m-1}\left(1-p_{i, m-1}\right)}
$$

4.更新模型 $F_{m}(x)$
$$
F_{m}(x)=F_{m-1}(x)+\nu \sum_{j=1}^{J_{m}} \gamma_{m}
$$

至此，通过多轮循环迭代， 就可以得到一个比较强的学习器 $F_m(x)$。

### 优缺点

优点：

回归树的生成过程可以理解成**自动进行多维度的特征组合**的过程，从根节点到叶子节点的完整路径，即**多个特征值的判断**
，才能决定一棵树的预测值；对**连续型特征值**的处理，GBDT 模型可以拆分出一个临界阈值 $\alpha$，例如   < $\alpha$ 的走左子树，>
$\alpha$ 的走右子树，**有效避免了人工离散化**，从而轻松解决了 LR 模型中**自动发现特征并进行有效组合**的问题。

缺点：

对**海量的 id 类特征**，GBDT 模型由于回归树的深度和数量限制（防止过拟合），不能实现有效存储；而且海量特征在也会存在性能瓶颈，当
GBDT 模型的 one-hot 特征维度大于 10 万时，就必须进行分布式的训练才能保证不爆内存。因此一般情况下 GBDT 模型会配合少量的反馈
CTR 特征来表达，这种做法虽然具有一定的范化能力，但同时会导致**信息损失**，无法有效地表达头部资源。

## 二、LR 模型

### 原理

LR 模型，即逻辑回归模型（$Logistics\space Regression$），是一种**基于回归分析的分类算法**。LR
模型与线性回归模型非常相似，区别在于线性回归模型处理的是数值问题，LR 模型则是使用 $Sigmoid$
函数将线性回归模型的分析结果转换为概率值。在推荐系统领域，与基于协同过滤的模型相比，LR
模型能够综合利用用户、物品、上下文等多种不同的特征生成较为“全面”的推荐结果。

### 算法

LR 模型的本质是在线性回归模型的基础上增加了 $Sigmoid$ 函数即非线性映射，使得 LR 模型成为了一个优秀的分类算法，其核心是 *
*LR 模型假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的**。

协同过滤和后续改进的矩阵分解都是利用用户间或者物品间的相似度实现推荐，LR
模型则将问题视作一个分类问题，通过预测正样本的概率对物品进行排序实现推荐，所谓的正样本指的是用户“点击”了某个商品或者“观看”了某个视频，均是推荐系统希望用户产生的“正反馈”行为，因此
**LR 模型将推荐问题转化为了一个点击率的预测问题**。点击率预测，站在用户的角度，仅为点击或未点击两种可能，是典型的二分类问题，非常适合用
LR 模型处理，具体过程如下：

1.将用户的年龄、性别，物品的属性、描述，当前时间、地点等**特征转换成数值型向量**

2.确定逻辑回归的优化目标，例如把点击率预测转换成二分类问题，进而可以将分类问题中常用的损失函数值作为优化目标，然后训练模型

3.模型训练完成后，在预测时，将特征向量输入模型产生预测，得到用户“点击”物品的概率

4.根据模型生成的“点击”概率对候选物品进行排序，产生推荐列表

处理过程如下图所示。

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200909215410263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:55%;" />
</div>

关键在于每个特征的权重参数 $w_i$，通常会使用梯度下降的方式，即首先随机初始化权重参数 $w_i$，然后将特征向量输入到 LR
模型，通过计算得到模型的预测概率，然后通过对目标函数求导得到 $w_i$ 的梯度，最后更新 $w_i$。

目标函数为：
$$
J(w)=-\frac{1}{m}\left(\sum_{i=1}^{m}\left(y^{i} \log f_{w}\left(x^{i}\right)+\left(1-y^{i}\right) \log \left(1-f_
{w}\left(x^{i}\right)\right)\right)\right.
$$
求导后为：
$$
w_{j}-\gamma \frac{1}{m} \sum_{i=1}^{m}\left(f_{w}\left(x^{i}\right)-y^{i}\right) x_{j}^{i} \rightarrow w_j
$$
通过多轮迭代就可以得到最佳的 $w_i$ 了。

### 优缺点

优点：

1.LR 模型**形式简单**，**可解释性强**，很直观地就能从特征的权重参数看到不同特征对最后推荐结果的影响；

2.训练模型时方便并行化，使用模型进行预测时只需要对特征进行线性加权，性能比较好，**适合处理海量 id 类特征**（id
类特征一个最重要的好处在于防止信息损失，对头部资源能有更细致的描述）；

3.资源占用少，尤其是内存。在实际工程应用中只需要存储权重比较大的特征及对应权重。

4.方便输出结果调整。LR 模型通过 $Sigmoid$ 函数输出的是每个样本的概率值，这使得我们很容易对这些概率值进一步划分阈值。

缺点：

1.表达能力不强，无法进行特征交叉，特征筛选等一系列需要有一定经验的训练人员才能进行的“高级”操作，可能会造成信息损失。

2.准确率不高。毕竟形式简单很难去拟合数据的真实分布。

3.处理非线性数据很麻烦。LR 模型仅能处理线性可分的数据，如果要处理非线性数据，则必须对具备连续特征的数据进行离散化，这又会带来多种问题。

4.LR 模型需要进行人工特征组合，这要求开发者必须具备非常丰富的领域经验，导致 LR 模型迁移困难，换一个领域就又需要重新进行大量特征工程。

## 三、GBDT + LR 模型

### 原理

通过分析 GBDT 模型和 LR 模型，我们可以发现：GBDT 模型可以自动发现特征并进行有效组合，但不适合处理海量 id 类特征；LR 模型可以处理海量
id 类特征，但需要人为发现特征并进行组合。可以得出结论，GBDT 模型和 LR 模型的优缺点非常适合互补。

2014 年，Facebook 提出了一种利用 GBDT 模型自动进行特征筛选和有效组合，生成新的离散特征向量，再把该特征向量当做 LR
模型的输入，产生最后的预测结果，即著名的 GBDT + LR 模型。目前，GBDT + LR 模型使用最广泛的场景是 CTR
点击率预估，即预测如果向用户推送广告，该广告会不会被用户点击。

### 算法

解释了 GBDT 模型和 LR 模型后，该模型也就非常好理解了：

<div align=center>
<img src="https://img-blog.csdnimg.cn/20200910161923481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:67%;" />    
</div>

**训练模型**阶段，GBDT 模型建立回归树的过程相当于自动进行特征组合和离散化，从根节点到叶子节点这条路径可以被视为不同特征进行的特征组合，用叶子节点可以唯一地表示这条路径，并将其作为离散特征传入
LR 模型进行**二次训练**。

**使用模型进行预测**阶段，数据会先从 GBDT 模型建立的回归树的根节点出发，到某个叶子节点对应的一个离散特征，即一组特征组合结束，然后把该特征以
one-hot 形式传入 LR 模型进行线性加权预测。

以模型图为例，设某条输入样本，从 GBDT 模型建立的回归树的根节点出发，到回归树的某个或某几个叶子节点处结束，生成了 LR
模型输入所需要的一维特征组合；例如，样本 x 经过左树，有三个叶子节点，出发后落在了左树的第二个节点，编码为 [0, 1, 0]
；落在了右树的第二个节点，编码为 [0, 1]；整体编码为 [0, 1, 0, 0, 1]，将该编码作为特征输入到 LR 模型进行分类。

### 关键点

1.**通过 GBDT 模型进行特征组合后产生的离散向量需要和训练数据的原特征一起作为 LR 模型的输入**，而不能仅用 GBDT 模型生成的离散特征。

2.GBDT 模型之所以属于集成学习算法，是因为一棵树的表达能力很弱，无法表达多个具备区分性的特征组合，而多棵树的表达能力更强。GBDT
模型在生成下一棵树之前都会学习当前树存在的不足，再开启新一轮的迭代，经过多少轮迭代就会生成多少棵树。

3.在 CRT 预测中，GBDT 模型一般会建立两类树，即一类非 ID 特征树，一类 ID 特征树，因为非 ID 和 ID 类特征在 CTR
预测中是非常重要的特征，直接作为特征建立树不可行，所以会为每个 ID 和非 ID 类特征建立 GBDT 回归树。

a.非 ID 类树：即不以细粒度的 ID 建立回归树，该类树作为 base，即便曝光少的广告和广告主，仍可以通过此类树得到有区分性的特征和特征组合。

b. ID 类树：即以细粒度的 ID 建立回归树，用于发现曝光充分的 ID 对应有区分性的特征和特征组合。

###  