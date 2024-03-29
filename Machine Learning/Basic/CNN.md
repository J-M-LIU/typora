# 卷积神经网络 Convolutional neural networks



## 从全连接层到卷积



### 不变性

​	**平移不变性(translation invariance)**: 不管检测对象出现在图像中的哪个位置，神经网络的前面几层 应该对相同的图像区域具有相似的反应；

​	**局部性(locality)**: 神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。



#### 卷积

数学中，两个函数 $g，f$之间的卷积定义为：(可看作图像函数 $f$ 和 核函数 $g$）
$$
(f*g)(\mathbf{x}) = \int g(\mathbf{z})f(\mathbf{x}-\mathbf{z})d\mathbf{z}
$$
也就是说，卷积是当把一个函数“翻转”并移位$x$时，测量f 和$g$之间的重叠.

离散对象中：
$$
(f*g)(i) = \sum_a g(a)f(i-a)
$$
对于二维张量：
$$
(f*g)(i,j) = \sum_a\sum_b g(a,b)f(i-a,j-b)
$$

#### 互相关运算 cross-correlation

卷积核的高度和宽度都是2

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220922233521867.png" alt="image-20220922233521867" style="zoom:50%;" />

​	卷积窗口从输入张量的左上⻆开始，从左到右、从上到下滑动。当卷积窗口滑动到新 一个位置时，包含在该窗口中的部分张量与卷积核张量进行按元素相乘。

输入长 X 宽：$n_h \times n_w$ ,卷积核长 X 宽：$k_h\times k_w$, 输出大小为：
$$
(n_h - k_h+1)\times (n_w - k_w + 1)
$$



### Padding

​	通常，如果我们添加$p_h$ 行填充(大约一半在顶部，一半在底部)和$p_w$ 列填充(左侧大约一半，右侧一半)，则输出形状将为
$$
(n_h −k_h +p_h +1)×(n_w −k_w +p_w +1)
$$
<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220922235600677.png" alt="image-20220922235600677" style="zoom:50%;" />

​	即输出高度和宽度分别增加 $p_h$和 $p_w$。

​	在许多情况下，我们需要设置$p_h = k_h − 1$和$p_w = k_w − 1$，使输入和输出具有相同的高度和宽度。这样可以在 构建网络时更容易地预测每个图层的输出形状。假设$k_h$ 是奇数，我们将在高度的两侧填充$p_h /2$行。如果$k_h$ 是 偶数，则一种可能性是在输入顶部填充<br>⌈$p_h /2$⌉行，在底部填充⌊$p_h /2$⌋行。同理，我们填充宽度的两侧。



### Stride

通常，当垂直步幅为$s_h$ 、水平步幅为$s_w$ 时，输出形状为:
$$
⌊(n_h −k_h +p_h +s_h)/s_h⌋×⌊(n_w −k_w +p_w +s_w)/s_w⌋
$$
在实践中，我们很少使用不一致的步幅或填充

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220923095020904.png" alt="image-20220923095020904" style="zoom:50%;" />

**通用公式**
$$
feature\ map\ size:((n −k +2* p)/s + 1)×((n −k +2* p)/s + 1)
$$
padding = "same" ，仅当stride=1时，padding = $(k-1)/2$



### 参数量计算

卷积参数量的计算对于一层卷积而言，其中的每个卷积核的参数量即为同卷积核的大小，那么对于有 $c_{out}$ 个卷积核的卷积层而言，并加上了偏置项。其参数量 $P$ 为：

$$
P=(k*k*c_{in}+1)*c_{out}
$$

### FLOPs计算

卷积Flops的计算衡量卷积计算量的指标是FLOPs (Floating Point Operations, 浮点运算次数) 。一次乘法和一次加法表示一个浮点运算次数。那么每一个卷积每一次的滑动卷积的计算量就是 $k*k*c_{in}$, 那么在$H*W$的图上进行卷积就可以计算得到如下计算公式，也就是参数量乘以卷积图的尺寸：
$$
FLOPs=P*H*W=(k*k*c_{in}+1)*c_{out}*H*W
$$

### 感受野的计算

感受野的计算感受野 (receptive Field) 是指卷积的输出结果对应于到输入中的区域大小。是设计多层卷积神经网络的一个重要依据。随着卷积层数的增加，对应到原始输入的感受野也会逐渐增大。层数 $i$ 对应到输入的感受野 $RF_{i}$ 的计算公式如下：

$$
RF_i=(RF_{i-1}-1)*s_i+k_i
$$

### 汇聚层 Pooling

池化层不包含参数，其运算是确定性的，通常计算汇聚窗口中所有元素的最大值或平均值。分别称为**最大汇聚层(maximum pooling)**和**平均汇聚层(average pooling)**。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220923101626049.png" alt="image-20220923101626049" style="zoom:50%;" />

### 输入通道与输出通道

输入通道指的是输入了几个二维信息，也就是很直观的rgb图有R、G、B三个通道，这决定了卷积核的通道数，即**输入图像的通道数决定了卷积核通道数**。

**输出通道是指卷积（关联）运算之后的输出通道数目，它决定了卷积核数量**，即需要输出通道数为几，就需要几个卷积核。

下图：输入一个三通道图像(3×7×7)，和两个三通道的卷积核(3×3×3)做了卷积运算，得到两个输出通道，即3×3×2的特征图。

<img src="https://segmentfault.com/img/bVW1tf?w=860&h=690" style="zoom:80%;" />

### 批量规范化

- 需要标准化输入特征 $\mathbf{X}$，使其平均值为0，方差为1。直观地说，这种标准化可以很好地与优化器配合使用，因为它可以将参数的量级进行统一。
- 随着训练学习率的补偿调整
- 更深层网络更容易过拟合，**正则化**。

​	每次训练迭代中，首先规范化输 入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。之后应用比例系数和比例偏移。即批量规范化。而选择合适大小的批量也很重要。
$$
\hat{\mu}_{\Beta} = \frac{1}{|\Beta|}\sum_{\mathbf{x}\in \Beta}\mathbf{x} \\
\hat{\sigma}_{\Beta} = \frac{1}{|\Beta|}\sum_{\mathbf{x}\in \Beta}(\mathbf{x}-\hat{\mu}_{\Beta})^2 + \epsilon
$$
批量规范化BN根据以下式子转换 $\mathbf{x}$ , $\gamma$-拉伸参数，$\beta$-偏移量。
$$
BN(\mathbf{x}) = \gamma ·\ \frac{\mathbf{x}-\hat{\mu}_{\Beta}}{\hat{\sigma}_{\Beta}} + \beta
$$


### 残差网络 ResNet

#### 为什么不能简单地增加网络层数

​	对于传统的CNN网络，简单的增加网络的深度，容易**导致梯度消失和爆炸**。针对梯度消失和爆炸的解决方法一般是**正则初始化(**normalized initialization**)**和**中间的正则化层(**intermediate normalization layers**)，**但是这会导致另一个问题，**退化问题**，随着网络层数的增加，在**训练集上的准确率却饱和甚至下降**了。这个和过拟合不一样，因为过拟合在训练集上的表现会更加出色。

​	按照常理更深层的网络结构的解空间是包括浅层的网络结构的解空间的，也就是说深层的网络结构能够得到更优的解，性能会比浅层网络更佳。但是实际上并非如此，深层网络无论从训练误差或是测试误差来看，都有可能比浅层误差更差，这也证明了并非是由于过拟合的原因。导致这个原因可能是因为**随机梯度下降的策略**，往往解到的并不是全局最优解，而是局部最优解，**由于深层网络的结构更加复杂，所以梯度下降算法得到局部最优解的可能性就会更大**。

#### 如何解决退化问题

既然深层网络相比于浅层网络具有退化问题，那么是否可以保留深层网络的深度，又可以有浅层网络的优势去避免退化问题呢？如果将深层网络的后面若干层学习成恒等映射 $h(x)=x$ ，那么模型就退化成浅层网络。但是直接去学习这个恒等映射是很困难的，那么就换一种方式，把网络设计成：
$$
H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x} \Longrightarrow F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}
$$
$\mathbf{x}$为输入，$F(\mathbf{x})$为经过线性变换和激活后的输出，

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221018092235296.png" alt="image-20221018092235296" style="zoom:50%;" />

[resnet 解析](https://zhuanlan.zhihu.com/p/72679537?utm_source=wechat_session)





## 局部连接和权值共享

### 权值共享

​	所谓权值共享就是说给定一张输入图片，用一个卷积核来卷积这张图，卷积核里的值叫做权重，整张图片在使用同一个卷积核内的参数。比如一个3×3×1的卷积核，这个卷积核内9个的参数被整张图共享，而不会因为图像内位置的不同而改变卷积核内的权系数。说的再直白一些，就是用一个卷积核不改变其内权系数的情况下卷积处理整张图片（当然CNN中每一层不会只有一个卷积核的，这样说只是为了方便解释而已）。

​	作用：大大减少网络训练参数的同时，还可以实现并行训练。

### 局部连接/稀疏连接

​	卷积层的节点仅仅和其前一层的部分节点相连接，只用来学习局部特征。局部感知结构的构思理念来源于动物视觉的皮层结构，其指的是动物视觉的神经元在感知外界物体的过程中起作用的只有一部分神经元。在计算机视觉中，图像中的某一块区域中，像素之间的相关性与像素之间的距离同样相关，距离较近的像素间相关性强，距离较远则相关性就比较弱，由此可见局部相关性理论也适用于计算机视觉的图像处理领域。因此，局部感知采用部分神经元接受图像信息，再通过综合全部的图像信息达到增强图像信息的目的。

​	从下图中可以看到，第n+1层的每个节点只与第n层的3个节点相连接，而非与前一层全部5个神经元节点相连，这样原本需要5×3= 5个权值参数，现在只需要3×3=9个权值参数，减少了40%的参数量，同样，第n+2层与第n+1层之间也用同样的连接方式。

<img src="https://img-blog.csdn.net/2018101716241781?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1bGluZ3dlbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:70%;" />
