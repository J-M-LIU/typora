# Batch Normalization



## 为什么要归一化

### 背景

​	对于中间的某一层，其前面的层可以看成是对输入的处理，后面的层可以看成是损失函数。一次反向传播过程会同时更新所有层的权重 $W_1$、$W_2$ ,..., $W_L$，前面层权重的更新会改变当前层输入的分布，而跟据反向传播的计算方式，我们知道，对 $W_k$ 的更新是在假定其输入不变的情况下进行的。如果假定第 $k$ 层的输入节点只有2个，对第 $k$ 层的某个输出节点而言，相当于一个线性模型 $y=w_1x_1+w_2x_2+b$，如图1所示。

<img src="https://img-blog.csdnimg.cn/20200429181244580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDAyMzY1OA==,size_16,color_FFFFFF,t_70" style="zoom:40%;" />

​	假定当前输入 $x_1$ 和 $x_2$ 的分布如图中圆点所示，本次更新的方向是将直线 $H_1$ 更新成 $H_2$，本以为切分得不错，但是当前面层的权重更新完毕，当前层输入的分布改变，直线相对输入分布的位置可能变成了 $H_3$，下一次更新又要根据新的分布重新调整。直线调整了位置，输入分布又在发生变化，直线再调整位置。对于浅层模型，比如SVM，输入特征的分布是固定的，即使拆分成不同的batch，每个batch的统计特性也是相近的，因此只需调整直线位置来适应输入分布，显然要容易得多。而深层模型，每层输入的分布和权重在同时变化，训练相对困难。

​	神经网络学习过程的本质就是为了学习数据分布，如果我们没有做归一化处理，那么每一批次训练数据的分布不一样，从大的方向上看，神经网络则需要在这多个分布中找到平衡点，从小的方向上看，由于每层网络输入数据分布在不断变化，这也会导致每层网络在找平衡点，显然，神经网络就很难收敛。

​	归一化后有什么好处呢？原因在于神经网络学习过程本质就是为了学习数据分布，一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降)，那么网络就要在每次迭代都去学习适应不同的分布，这样将会大大降低网络的训练速度。

​	深度网络的训练是一个复杂的过程，只要网络的前面几层发生微小的改变，那么后面几层就会被累积放大下去。一旦网络某一层的输入数据的分布发生改变，那么这一层网络就需要去适应学习这个新的数据分布，所以如果训练过程中，训练数据的分布一直在发生变化，那么将会影响网络的训练速度。

### 标准差归一化

​	如何解决训练过程中间层数据分布发生改变的情况？对每一维数据进行标准化，即使每一维特征值均值为0、方差为1。$x_i^{(b)}$ 表示第 $b$ 个样本中第 $i$ 个特征。具体训练过程中表示：训练过程中采用batch 随机梯度下降。
$$
\hat{x}_i^{(b)} = \frac{x_i^{(b)}-E[x_i]}{\sqrt{Var[x_i]}}
$$

## BN层

### 定义

​	与激活函数层、卷积层、全连接层、池化层一样， BN层也属于CNN中网络的一层。BN本质原理上是将数据归一化至均值0、方差为1；BN层是可学习的，参数为γ、β的网络层，它是针对一个batch里的数据进行归一化和尺度操作，且一旦神经网络训练完成后，BN的尺度参数也固定了，这就是一个完全变成一个关于归一化数据的仿射变换。


### 算法概述

​	为什么不能简单地实现每一层中间层的标准差归一化呢？如果是仅仅使用上面的归一化公式，对网络某一层A的输出数据做归一化，然后送入网络下一层B，会影响到本层网络A所学习到的特征。比如，网络中间某一层学习到的特征数据本身就分布在S型激活函数的两侧，强制将其归一化处理、标准差也限制在了1，把数据变换成分布于s函数的中间部分，这样就相当于这一层网络所学习到的特征分布被破坏了。

​	因此引入**可学习的参数 $\gamma,\beta$** ，首先对批训练数据标准化，之后对标准化数据进行缩放和移位到新的分布 $y$，具有新的均值 $\beta$ 和方差 $\gamma$。
$$
\begin{array}{l}
\text { Input: Values of } x \text { over a mini-batch: } \mathcal{B}=\left\{x^{(1) \ldots (m)}\right\} \text {; } \\
\text { Parameters to be learned: } \gamma, \beta \\
\text { Output: }\left\{y^{(i)}=\mathrm{BN}_{\gamma, \beta}\left(x^{(i)}\right)\right\} \\
\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \quad \quad / / \text { mini-batch mean } \\
\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x^{(i)}-\mu_{\mathcal{B}}\right)^{2} \quad \text { // mini-batch variance } \\
\widehat{x}^{(i)} \leftarrow \frac{x^{(i)}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \quad \text { // normalize } \\
y^{(i)} \leftarrow \gamma \widehat{x}^{(i)}+\beta \equiv \mathrm{BN}_{\gamma, \beta}\left(x^{(i)}\right) \quad \text { // scale and shift } \\
\end{array}
$$

​	假设BN层有 n 个输入节点，则 $X$ 可构成 $n\times m$ 大小的矩阵，BN层相当于通过行操作将其映射为另一个 $n\times m$ 大小的矩阵，如图2所示。

<img src="https://img-blog.csdnimg.cn/20200429181926282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDAyMzY1OA==,size_16,color_FFFFFF,t_70" style="zoom:60%;" />

​	将标准化过程和缩放位移写到一个公式里，其中，$x_i^{(b)}$ 表示当前输入batch中第 $b$ 个样本的第 $i$ 个输入节点的值，$\epsilon$ 为防止除零引入的极小量(可忽略). 每一个输入输入节点/神经元均具有两个可学习参数 $\beta$ 和 $\gamma$，当 $\beta = E[x_i]\ and \ \gamma = \sqrt{Var[x_i]}$ 时，可以恢复出原始某一层学到的特征。
$$
y_i^{(b)}  \equiv \mathrm{BN}_{\gamma, \beta}\left(x_i^{(b)}\right) = \gamma ·\left(\frac{x_i^{(b)}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} \right) + \beta
$$
**位置**

原paper建议将BN层放置在ReLU前，因为ReLU激活函数的输出非负，不能近似为高斯分布。

### 优点

​	BN层归一化数据接近于高斯分布，解决了训练时候中间层数据分布发生改变的问题。而且BN层允许选择更大的学习率，可以加速训练。此外，BN解决了深度梯度消失和梯度爆炸的问题；BN还可以改善正则化策略：作为正则化的一种形式，轻微减少了对dropout的需求，不用管过拟合中drop out、L2正则项参数的选择问题。
### 缺点

​	BN层每次处理的批量的均值和标准差都不同，所以这相当于加入了噪声，增强了模型的泛化能力。但对于图像超分辨率重建、图像生成、图像去噪和图像压缩等生成模型并不友好，生成的图像要求尽可能清晰，不应该引入噪声。

​	当batch size越小，BN的表现效果也越不好（batch>32），因为计算过程中所得均值和方差不能代表全局；之后针对这个问题，何恺明团队提出了**GN（group normalization）**，先把通道C分成G组，然后单独拿出来归一化，最后把G组归一化之后的数据合并成CHW，解决了batch很小时BN效果不好的问题。如图3所示。

<img src="https://img-blog.csdnimg.cn/878bae1c5db0415ea1ef83b0f82729d7.png" style="zoom:55%;" />
