# Lightweight Image Super-Resolution with Information Multi-distillation Network

# IMDN

超分辨网络结构：特征提取（convs）、非线形变换（残差块）、图像重构（上采样+convs）如图

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231025150016436.png" alt="image-20231025150016436" style="zoom:50%;" />

## Motivation

1. 轻量级的超分网络，作者探讨了一些较为（训练）快速的方法（全局残差学习（VDSR）和较高的初始学习率，递归学习（DRCN）大幅减少了参数量，递归学习（DRRN），MemNet通过将多个memory block用dense connection的方式连解决了CNN架构中的long-term dependency的问题。）。作者认为，为了加速超分重建，最好是对LR进行下采样，**在低分辨率的尺寸下进行重建过程。**

2. 实现单模型任意尺度的超分重建，不同于meta-SR的思路，作者将任意尺度超分建模为 上采样+ densen prediction 的问题，先对LR上采样再通过一个全卷积网络，这样输入模糊输出清晰。但是这样有一个很致命的问题，**全程都是在HR尺寸下进行，计算量很大**。

3. 为了能使本文模型（**在小尺寸下进行，计算量比较小**）也能完成任意尺度超分，作者提出（Adaptive cropping strategy，ACS）策略，对输入进行padding确保能够被比例因子整除。（个人认为任意尺度超分并不是本文的主要创新点）

## Introdutcion

对于SR网络，过多的卷积会限制超分辨率技术在低算力设备上的应用，当前的主流方法趋势是加深残差块深度；并且以往的方法没有解决任意尺度因子的超分辨率问题。本文提出轻量级的信息多蒸馏网络(IMDN)，其中包含蒸馏和选择性融合部分，结合了信息多蒸馏块 (IMDB) 和对比感知通道注意机制 (CCA)。蒸馏模块分步提取分层特征，融合模块通过对比感知通道注意力机制进行评估候选特征重要性，然后根据重要性聚合特征。为了处理任意大小的真实图像，本文提出了一种自适应裁剪策略(ACS)。

本文贡献（Contributions）

（1）提出了轻量级的信息多重蒸馏网络（IMDN）以及它的基本组成块（IMDB）

（2）提出了基于对比度的通道注意力（ Contrast-aware channel attention(CCA) ）

（3）提出了自适应裁剪策略（ Adaptive cropping strategy(ACS) ）



## METHOD



![image-20231025144915917](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231025144915917.png)

### 信息多重蒸馏网络 (IMDN)

IMDN网络结构如 Figure 1 所示。首先通过3x3的卷积进行LR图像的浅层特征提取，然后就到了整个网络的关键组件，即一系列堆叠的IMDB进行深层特征提取。然后通过1x1的卷积将各个IMDB的output融合到一起，再经过一层卷积后，通过上采样得到SR图像。所以整个网络结构可以概括为：浅层特征提取、深层特征提取、特征融合、上采样重建。整个网络的重点就是深度特征提取部分的IMDB结构。

给定一个低分辨率(LR)的输入图像$I^{LR}$, 其对应的高分辨率(HR)目标图像为$I^{HR}$。利用这个网络，可以产生一个超分辨率(SR)的图像 $I^{SR}$, 公式如下：

$I^{SR}=H_{IMDN}(I^{LR})$  其中$H_{IMDN}(\cdot)$代表IMDN网络。为了优化网络，使用了均值绝对误差(MAE)损失，这与许多之前的工作相一致。给定一个训练集，其中包含$N$ 对LR-HR图像对，IMDN的损失函数可以表示为：
$$
L(\theta)=\frac1N\sum_{i=1}^N\|H_{IMDN}(I_i^{LR})-I_i^{HR}\|_1
$$
其中$\theta$表示模型的可更新参数，$\|\cdot\|_1$ 代表L1范数。



### Information multi-distillation block IMDB

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231025151212307.png" alt="image-20231025151212307" style="zoom:40%;" />

如Figure 2所示，IMDB 由渐进细化模块PRM、对比感知通道注意力(CCA)层和用于减少特征通道数量的1 × 1卷积构建。整块采用残差连接。该块的主要思想是像DenseNet[9]一样一点点提取有用的特征。

PRM 首先采用3 × 3卷积层提取输入特征，用于后续的多个蒸馏(细化)步骤。对于每一步，对上述特征进行通道拆分操作，将产生两部分特征。经过细化的特征被保留，另一部分被输入到下一个计算单元。

具体地说，给定输入特征 $F_{in}$，此过程在第 n 个IMDB中可以描述为：
$$
F_{refined\_j}^n,F_{coarse\_j}^n=Split_j(CL_j(F_{in}^n))
$$
其中$CL_j$表示第j个卷积层 (包括Leaky ReLU), 而$Split_j$ 表示第j个通道拆分层.

之后将每个阶段细化得到的特征 concat起来：
$$
\begin{gathered}
F_{\boldsymbol{distilled}}^n= 
Concat\left(F_{refined\_1}^n,F_{refined\_2}^n,F_{refined\_3}^n,F_{refined\_4}^n\right)
\end{gathered}
$$
**对比感知通道注意力**

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231025151232427.png" alt="image-20231025151232427" style="zoom:40%;" />

全局最大/平均池化主要是从 high-level或 mid-level 中捕获全局信息。虽然平均池化可以增加PSNR值，但它缺乏关于结构、纹理和边缘的信息，而这些都是提高SSIM所需的。SR更多的考虑图像纹理、边缘等low-level的信息，所以作者选择了一种基于对比度的方法来分配权重。具体实现上，计算每层feature map的均值和标准偏差，以两者之和作为对比度信息送入CCA。

为了计算对比度信息值，给定特征映射 $X=[x_1,x_2,...,x_C]$, 我们可以得到：

$$
\begin{aligned}
\text{zc}& =H_{GC}\left(x_{c}\right)  \\
&=\sqrt{\frac1{HW}\sum_{(i,j)\in x_c}\left(x_c^{i,j}-\frac1{HW}\sum_{(i,j)\in x_c}x_c^{i,j}\right)^2}+ \\
&\frac1{HW}\sum_{(i,j)\in x_c}x_c^{i,j},
\end{aligned}
$$
 其中 $z_c$ 是第c个元素的输出，而 $H_{GC}(c)$ 表示对比度信息评估函数。
 最后，为了在这个块中提取更有用的信息，我们采用了类似DenseNet的残差连接方法。



### Adaptive cropping strategy

ACS的目的在于实现任意尺度超分，前面提到，在HR尺寸下进行运算计算量是非常大的。所以作者想要在LR尺度进行任意尺度超分。

作者在文中为了说明ACS的作用，提出了一个IMDN_AS进行说明。IMDN_AS与IMDN的区别，就是将IMDN前面的卷积层换成了两个下采样的卷积层加上stride，作者代码中 s2=2 ，即两个卷积代表（x4）的下采样，这样做的目的是减少参数量（后面的运算都在小尺寸下进行），后面的Upsampler再进行一个（x4）的上采样，这样，输出的尺寸就和输入一样了。

为了确保输入能进行（x4）下采样，需要保证输入的宽高能被4整除，因此IMDN_AS引入了ACS的策略，ACS就是保证了输入的宽高能被尺度因子整除。ACS对输入LR引入了padding，确保输入的LR可以对任意尺度因子进行下采样，padding由一个超参数进行控制。