# Multilayer Perceptron

## 在网络中加入隐藏层

通过在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型。



<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920104010534.png"  style="zoom:40%;" />

### 从线性到非线性

假设有n个样本，d个输入特征，q个输出类别，隐藏层中有h个隐藏单元。

隐藏层权重：$W^{(1)}\in R^{d\times h}$, 偏置：$b^{(1)} \in 1\times h $

输出层权重：$W^{(2)}\in R^{h\times q}$, 偏置：$b^{(2)} \in 1\times q $
$$
\begin{gathered}
H = XW^{(1)}+b^{(1)}\\
O = HW^{(2)}+b^{(2)}
\end{gathered}
$$
合并隐藏层后，不难发现叠加多层线性模型，最后仍得到的是等价的线性模型；仿射函数的仿射函数本身就是仿射函数，但是我们之前的 线性模型已经能够表示任何仿射函数：

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920115301105.png" style="zoom:50%;" />

为了发挥多层架构的潜力，我们还需要一个额外的关键要素:在仿射变换之后对每个隐藏单元应用非线性的激活函数(activation function) $\sigma$. 有了激活函数，就不可能再将我们的多层感知机退化成线性模型.
$$
\begin{gathered}
H^{(1)} = \sigma_1 (XW^{(1)}+b^{(1)})\\
H^{(2)} = \sigma_2 (H^{(1)}W^{(2)}+b^{(2)})\\
O = H^{(2)}W^{(3)}+b^{(3)}
\end{gathered}
$$



### 激活函数

激活函数(activation function)通过计算加权和并加上偏置来确定神经元是否应该被激活，它们将输入信号转换为输出的可微运算。大多数激活函数都是非线性的。

#### ReLU函数：Rectified linear unit

Relu函数提供了简单的非线性变换，给定元素 $x$，ReLU函数被定义为元素x与0的最大值：
$$
ReLU(x) = max(x,0)
$$
通俗地说，ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920164639979.png" alt="image-20220920164639979" style="zoom:50%;" />

##### 导数

当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。注意，当输入值精确等于0时， ReLU函数不可导。在此时，我们默认使用左侧的导数，即当输入为0时导数为0。我们可以忽略这种情况，因为输入可能永远都不会是0。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920164943723.png" alt="image-20220920164943723" style="zoom:50%;" />

**使用ReLU的原因**：**它求导表现得特别好:要么让参数消失，要么让参数通过。这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题。**



#### sigmoid函数

对于一个定义域在$R$中的输入，sigmoid将输入压缩变换为区间(0,1)上的输出。可将sigmoid看作softmax的特例。
$$
sigmoid(x) = \frac{1}{1+exp(-x)}
$$
<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920165940322.png" alt="image-20220920165940322" style="zoom:50%;" />

- 当输入接近0，sigmoid函数接近线性变换
- 目前基本使用ReLU代替sigmoid

##### 导数

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920170135578.png" alt="image-20220920170135578" style="zoom:50%;" />



#### tanh函数

将输入压缩变换到区间(-1,1)上。
$$
tanh(x) = \frac{1-exp(-2x)}{1+exp(-2x)}
$$

- 当输入在0附近时，tanh函数接近线性变换。函数的形状类似于sigmoid函数， 不同的是tanh函数关于坐标系原点中心对称。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920170407073.png" alt="image-20220920170407073" style="zoom:50%;" />



### 模型选择、欠拟合和过拟合

#### 影响模型泛化的因素

1. 模型的简单性：模型的维度（高维、低维）/权重参数的个数/权重参数的范数。
2. 参数采用的值。当权重的取值范围较大时，模型可能更容易过拟合。
3. 训练样本数据过少，模型容易过拟合。

**如何发现可以泛化到模式是机器学习的根本问题。**

- 困难在于，当我们训练模型时，我们只能访问数据中的小部分样本。最大的公开图像数据集包含大约一百万张图像。而在大部分时候，我们只能从数千或数万个数据样本中学习。在大型医院系统中，我们可能会访问数十万份医疗记录。当我们使用有限的样本时，可能会遇到这样的问题:当收集到更多的数据时，会发现之前找到的明显关系并不成立。

- 模型在训练数据上拟合的比潜在分布中更接近的现象称为过拟合(overfitting), 用于对抗过拟合：（1）==正则化==(regularization). <u>当训练误差明显低于验证误差表明过拟合</u>；（2）==增加训练数据数量==。
- 训练误差(training error): 模型在训练数据集上计算得到的误差。
- 泛化误差(generalization error): 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。

#### K折交叉验证

将原始训练数据分为K个不重叠的子集。执行K次模型训练和验证。每次在K-1个子集上进行训练，并在剩余的一个子集上进行验证。最后对K次实验的结果取平均来获取训练和验证误差。

#### 模型复杂度对拟合的影响

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220920205256382.png" alt="image-20220920205256382" style="zoom:50%;" />



### 权重衰减

#### 范数

$L_1$范数：向量元素的绝对值之和
$$
||X||_1 = \sum_{i=1}^n|x_i|
$$
$L_2$范数
$$
||X||_2 = \sqrt{\sum_{i=1}^nx_i^2}
$$
欧几里得距离是一个 $L_2$范数。

**一般的，$L_p$范数：**
$$
||X||_p = (\sum_{i=1}^n|x_i|^p)^{1/p}
$$



#### $L_2$正则化——岭回归 Ridge Regression

​	正则化的含义是：在训练集的损失函数中加入惩罚项，以降低学习到的模型的复杂度。可通过函数与零的距离来衡量函数的复杂度；如线性函数 $f(x)=w^Tx$中，通过权重向量 $||w||^2$来度量其复杂性。要保证权重向量比较小，最常用方法是将训练目标最小化训练标签上的预测损失，调整为最小化预测损失与惩罚项之和。
$$
L(W,b)=\frac{1}{2n}\sum_{i=0}^n(W^TX^{(i)}+b-y^{(i)}) + \frac{\lambda}{2}||w||^2
$$
**小批量随机梯度下降**
$$
W:=(1-\alpha\lambda)W-\frac{\alpha}{\Beta}\sum_{i\in\beta}X^{(i)}(W^TX^{(i)}+b-y^{(i)})
$$



### 前向传播、反向传播和计算图



#### 前向传播 forward propagation

按顺序(从输入层到输出层)计算和存储神经网络中每层的结果。

假设输入样本是 $\mathbf{x}\in R^d$, 且隐藏层不包括偏置, 隐藏层权重参数$W^{(1)}\in R^{d\times h}$
$$
\mathbf{z} = W^{(1)}\mathbf{x}
$$

将中间变量$\mathbf{z} \in R^h$通过激活函数φ后，我们得到⻓度为h的隐藏激活向量:
$$
\mathbf{h} = \phi{(\mathbf{z})}
$$
假设输出层的权重 $W^{(2)}\in R^{q\times h}$, 得到输出层长度为 $q$的向量：
$$
\mathbf{o} = W^{(2)}\mathbf{h}
$$
假设损失函数为 $l$，样本标签为 $y$，单个数据样本的损失项为：
$$
L = l(\mathbf{o}, y)
$$
给定超参数 $\lambda$, 正则化项为：
$$
s = \frac{\lambda}{2}(||W^{(1)}||^2 +||W^{(2)}||^2 )
$$
正则化损失为：
$$
J = L + s
$$



#### 反向传播 backpropagation

​	指的是计算神经网络参数梯度的方法，根据微积分中的链式规则，按相反的顺序从输出层到输入层遍历网络。存储了计算某些参数梯度时所需的任何中间变量(偏导数)。前向传播通过训练数据和权重参数计算输出结果；反向传播通过导数链式法则计算损失函数对各参数的梯度，并根据梯度进行参数的更新。

​	如上节前向传播中举例，单隐藏层网络的参数为 $W^{(2)}$ 和 $W^{(2)}$，反向传播的目的是计算梯度 $\frac {\partial J}{\partial W^{(1)}}$ 和  $\frac {\partial J}{\partial W^{(2)}}$，计算的顺序与前向传播相反。
$$
\frac {\partial J}{\partial L} = 1 \ and \  \frac {\partial J}{\partial s} = 1
$$

$$
\frac {\partial J}{\partial \mathbf{o}} = prod(\frac {\partial J}{\partial L},\frac {\partial L}{\partial \mathbf{o}}) = \frac {\partial L}{\partial \mathbf{o}}
$$

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230130113358285.png" alt="image-20230130113358285" style="zoom:50%;" />

> 
