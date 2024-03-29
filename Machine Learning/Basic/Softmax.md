## softmax回归

### 特征

特征一般分为两类：数值特征和离散特征。数值特征是指特征本来就是一个有意义的数，往往可以直接作为模型的输入，比如人的年龄，物品的大小，重量等，对于大部分数值特征来讲，处理方法比较简单，往往只需要正则化把数值映射到某个范围即可，比如通过标准正态分布计算标准分。离散特征是指那些不能用一个有意义的数来代表的特征，比如人的性别，人的国籍，英文单词，推荐系统中的物品id和用户id等等。这些离散特征是没有办法直接输入到模型的，那么如何在输入模型之间处理这些特征，让模型能够理解这些特征具体是什么呢？

#### 离散特征处理

离散特征处理一般分两步，第一步是建立字典，把类别映射成序号。比如性别，可以把’男‘映射成0，‘女’映射成1，或者国籍，可以把‘中国’映射成0，‘美国’映射成1，‘俄罗斯’映射成2，等等。

第二步是向量化，即把之间映射的序号再一次映射，变成向量。向量化有两个办法：

##### embedding：把序号映射成低维的稠密向量。

Embedding的原理其实是用一个参数矩阵乘以one hot编码。举个例子，如果一个特征有20个类别，我们希望用一个5维的[embedding](https://www.zhihu.com/search?q=embedding&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2637307981})来表示这个特征。那么我们需要一个 $5\times20$ 的参数矩阵，用这个矩阵把这个特征20维的one hot编码投影到5维空间里面。这个参数矩阵有很多训练方法，一个比较好的embedding可以很好的描述特征类别之间的关系，比如电影中，同类别的电影所对应的embedding向量会很接近。

现在有很多公开的embedding模型，比如自然语言处理中的**BERT**，BERT用wiki data的数据为英文单词训练了一套pretrianed embedding，现在在工业界非常常用。很多场景当中把这个训练好的embedding拿来在自己的数据集fine tune一下即可立刻投入使用。

##### One hot编码：把序号映射成高维的稀疏向量。

独热编码是一个向量，它的分量和类别一样多。类别对应的分量设置为1，其他所有分量设置为0。标签y将是一个三维向量，其中(1, 0, 0)对应于“猫”、(0, 1, 0)对应于“鸡”、(0, 0, 1)对应于“狗”:
$$
y \in \{(1,0,0),(0,1,0),(0,0,1)\}
$$
为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。假设输入为2 x 2的图像，各个像素值分别对应4个特征 $x_1,x_2,x_3,x_4$. 则有4个特征和3个可能的输出类别。
$$
\begin{aligned}
o_1 = w_{11}x_1+w_{12}x_2+w_{13}x_3 + w_{13}x_4 + b_1\\
o_2 = w_{21}x_1+w_{22}x_2+w_{23}x_3 + w_{24}x_4 + b_2\\
o_3 = w_{31}x_1+w_{32}x_2+w_{33}x_3 + w_{34}x_4 + b_3
\end{aligned}
$$
与线性回归一样，softmax回归也是单层的神经网络，由于计算每个 $o_1,o_2,o_3$ 需要所有输入 $x_1,x_2,x_3,x_4$ , 所以softmax回归的输出层也是全连接层。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220916223701708.png" alt="image-20220916223701708" style="zoom:50%;" />

#### softmax运算

softmax：非线性函数
$$
\hat{y} = softmax(o) ，其中 \hat{y}_j = \frac{exp(o_j)}{\sum_kexp(o_k)}
$$

尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。因此，softmax回归是一个线性模型(linear model)。

#### 小批量样本的矢量化

读取样本 $X$，其中特征数量为$d$，批量大小为$n$，假设输出有$q$个类别，则网络输出有$q$个类别，那么小批量样本的特征为$X\in R^{n\times d}$, 权重为 $W \in R^{d\times q}$ , 偏置为 $b\in R^{1\times q}$ ，输出为 $O \in R^{n\times q}$, $O_{nq}$的一行代表一个样本的权重输出，预测值$\hat{Y}\in R^{n\times q}$，$\hat{Y}_{n\times q}$的一行代表一个样本在各个类别的预测概率值，共有$n$个样本。softmax回归的矢量计算表达式为:

$$
O = XW+b\\
\hat{Y}=softmax(O) = softmax(XW+b)
$$

 $X$中一行代表一个样本，每一行为d个输入特征.

**实现softmax的步骤**

- 对 $O_{nq}$中每一项求幂；
- 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数$\sum_{k=1}^qexp(o_{jk}) \  (j=1,2,...,n)$
- 将每一行除以其规范化常数，确保结果的和为1



#### **交叉熵损失函数**

1. 二分类

在二分的情况下，模型最后需要预测的结果只有两种情况，对于每个类别我们的预测得到的概率为 $p$和 $1-p$，根据极大似然估计，表达式为：

> $\hat{y_i}$是样本 $i$ 预测为正类的概率
>
> $p_i$表示样本 $i$ 的label，正类为 1 ，负类为0


$$
P(Y|X) = \prod_{i=1}^n(\hat{y_i}^{p_i} \times (1-\hat{y_i})^{1-p_i})
$$
引入log函数，因为不会改变函数的单调性；由于尽量使得 $P(Y|X)$更大，即使得其负数更小：
$$
L = \frac{1}{N}\sum_{i=1}^nL_i = -\frac{1}{N} \sum_{i=1}^n [p_i ·log(\hat{y_i}) + (1-p_i)·log(1-\hat{y_i})]
$$

2. 多分类

多分类的情况就是对二分类进行拓展
$$
P(Y|X) = \prod_{i=1}^nP(\hat{y}^{(i)}|x^{(i)})=  \prod_{i=1}^n{(\hat{y}^{(i)})^{y^{(i)}}}
$$

> $y_i$表示样本 $i$ 的label，正类为 1 ，负类为0

对数化得损失函数为

> $q$ :分类类别数
>
> $n$ : 样本数

$$
-logP(X|Y) = -\sum_{i=1}^n logP(\hat{y}^{(i)}|x^{(i)}) = -\sum_{i=1}^n L(y^{(i)},\hat{y}^{(i)})
$$

一个样本对应的损失函数为：

$$
L(y,\hat{y}) = - \sum_{j=1}^q y_{j}log\hat{y}_j
$$

#### 梯度下降

$$
\begin{gathered}
L(y,\hat{y}) = -\sum_{j=1}^q y_j log \frac{exp(o_j)}{\sum_{k=1}^q exp(o_k)}\\
=\sum_{k=1}^q y_j log{\sum_{k=1}^q exp(o_k)} - \sum_{j=1}^q y_jo_j\\
=log{\sum_{k=1}^q exp(o_k)} - \sum_{j=1}^q y_jo_j

\end{gathered}
$$

对于 $o_j$ 求导得到：
$$
\frac{\partial{L(y_j,\hat{y_j})}}{\partial{o_j}} = L'(y,\hat{y}) =\frac{exp(o_j)}{\sum_{k=1}^q exp(o_k)} - y_j = softmax(o_j) - y_j
$$
**参数更新**
$$
\frac{\partial{L}}{\partial{w_{ij}}} = \frac{\partial{L}}{\partial{o_j}} \frac{\partial{o_j}}{\partial{w_{ij}}} = (softmax(o_j) - y_j)·x_i   \\i \in[1,d],j\in[1,q]
$$
换句话说，导数是我们softmax模型分配的概率与实际发生的情况(由独热标签向量表示)之间的差异。从这个意义上讲，这与我们在回归中看到的非常相似，其中梯度是观测值 $y$ 和估计值 $\hat{y}$ 之间的差异。

## 交叉熵

对于两个概率分布，一般可以用交叉熵来衡量它们的差异。标签的真实分布 𝒚 和模型预测分布 𝑓(𝒙; 𝜃) 之间的交叉熵为：
$$
L(\mathbf{y},f(\mathbf{x},\theta)) = -\mathbf{y}^T \log f(\mathbf{x},\theta)\\
=-\sum_{i=1}^{n}y_i \log f_i(\mathbf{x},\theta)
$$

其中，$\mathbf{y}$ 为标签的真实概率分布，$f(\mathbf{x},\theta)$ 为预测概率分布。因为 $\mathbf{y}$ 为one-hot向量，因此也可写作：
$$
L(\mathbf{y},f(\mathbf{x},\theta)) = -\log f_y(\mathbf{x},\theta)
$$
其中 $f_y(\mathbf(x),\theta)$ 可以看作真实类别 $\mathbf{y}$ 的似然函数。因此，交叉熵损失函数也就是**负对数似然函数**

