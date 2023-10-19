# DCVC: Deep Contextual Video Compression



## 简介

​	在本文中，提出了一个深度上下文(deep contextual)视频压缩框架，以实现从预测编码到条件编码的范式转换。
​	
​	已知，残差编码的信息熵 大于等于 条件编码的信息熵：$H(x_t - \bar{x_t}) \geq H(x_t|\bar{x_t})$；
​	已有的基于 Auto-Encoder的图像压缩，可通过 autoencoder 探索图像间的关系；本文提出构建基于条件编码的自编码器AE，代替残差编码。

## 相关工作

​	现有的深度视频压缩工作可分为非时延约束和时延约束两类。
​	非时延约束：参考帧可以来自未来。[Video compression through image interpolation,]中，对之前的帧和未来帧进行帧插值得到预测帧，然后进行残差编码；[Neural inter-frame compression for video coding.]则引入光流运动估计。这种编码方式会带来更大的延迟，并且会显著增加GPU的内存成本。
​	低时延约束：参考帧仅来自前一帧。DVC[DVC: an end-to-end deep video compression framework.]将传统视频编解码的模块均替换为network；[Improving deep video compression by resolutionadaptive ﬂow coding.]编码运动矢量时考虑 率失真优化。

## Proposed Method

### Framework of DCVC

**传统视频压缩 残差编码**
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t− \tilde{x}_t) ⌉ ) + \tilde{x}_t \ with \   \tilde{x}_t= f_{predict}(\hat{x}_{t-1}).
$$

**条件编码**
	一种简单直接的条件编码方式是直接使用预测帧 $\tilde{x}_t$ 作为条件：
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t|\tilde{x}_t) ⌉ \ | \ \tilde{x}_t )  \ with \   \tilde{x}_t= f_{predict}(\hat{x}_{t-1}).
$$
​	但预测帧只包含 RGB 三个通道的像素信息，这会限制条件编码的潜力，所以可以让模型自己学习所需要的条件。在 DCVC 框架中, 时域高维上下文特征为编码条件。相比传统的像素域的预测帧，高维特征可以提供更丰富的时域信息，不同的通道也可以有很大的自由度去提取不同类型的信息，从而帮助当前帧高频细节获得更好的重建。
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t|\bar{x}_t) ⌉ \ | \ \bar{x}_t )  \ with \   \bar{x}_t= f_{context}(\hat{x}_{t-1}).
$$
**Notations**
$x_t$: current frame.
$\tilde{x}_t$: predicted frame.
$\bar{x}_t$ : context.
$y_t$: latent code.
$\hat{y}_t$: quantized $y_t$ (通过round操作)

传到解码器端: 
运动信息(quant_mv), 运动先验信息(compressed_z_mv);
context信息(compressed_y_renorm), context先验信息(compressed_z)

对当前帧的encode和decode都基于 context值 $\bar{x}_t$，而 $x_t$ 与 $\bar{x}_t$ 的关系通过网络自适应地学习得到，而非以往通过相减操作得到残差 $r_t = x_t - \bar{x}_t$。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221101162222345.png" alt="image-20221101162222345" style="zoom:40%;" />

**具体流程为**：

1. 通过光流估计网络得到上一重构帧 $\hat{x}_{t-1}$ 到当前帧 $x_t$ 的运动信息 $m_t$，然后encode得到编码MV $g_t$，并发送至解码端；
2. 上一重构帧 $\hat{x}_{t-1}$ 通过特征提取网络从 pixel domain 转换到 feature domain，并与解码后到重构运动向量 $\hat{m}_t$ 进行 $warp(\hat{x}_{t-1},\hat{m}_t)$ 操作，并通过特征精调网络 context refinement 获取最终的context值 $\bar{x}_t$，对 $\bar{x}_t$ 通过 概率模型(Entropy Model)编码后，输入到解码端。**（之前只是获得 $\hat{x}_{t-1}$ 和 $\bar{x}_{t}$ 之间的残差，残差编码假设帧间预测总是最有效的，但这种假设可能不足以处理新的内容。上下文是在具有更高维度的特征域中的。这种设计允许不同的通道自由地提取不同类型的信息。例如，某些通道可能更加关注高频内容，而其他通道可能关注特定的颜色信息。这些各种上下文特征使得DCVC能够实现更好的重建质量，特别是对于包含大量高频的复杂纹理。）**


对 latent code $y_t$ 进行量化和算数编码操作，运动信息 $m_t$ 通过MV Encoder压缩后均传送至解码器端。

**个人理解：contextual encoder的作用类比于DVC中获取残差的过程，context $\bar{x}_t$ 也是通过warp操作，即运动补偿操作获取的，作用类似于DVC中的运动补偿帧。** 



### Entropy Model

**交叉熵与实际比特率的关系**
$$
R(\hat{y}_t)\geq E_{y_t\sim q\hat{y}}[-\log_2p_{\hat{y}}(\hat{y}_t)]
$$
 这个公式描述了交叉熵与实际比特率之间的关系。
 $R(\hat{y}_t)$ 代表的是实际比特率。
 $p_{\hat{y}}(\hat{y}_t)$ 和 $q_{\hat{y}}(\hat{y}_t)$ 分别是量化latent codes的预测和真实概率分布。
 $E_{yt\sim q\hat{y}}$ 表示期望是在真实概率分布 $q_{\hat{y}}$ 下进行的。

公式的意思是说, 基于预测的概率分布, 交叉熵是实际比特率的下限。理想情况下, 如果能够准确估计概率分布, 交叉熵与实际比特率之间的差距会非常小。

**Laplace分布**
$$
p_{\hat{y}}(\hat{y}_t|z_t)=\Pi_i[C(\mu_{i,t},\sigma_{i,t}^2)*U(-\frac{1}{2})^{\hat{y}_{i,t}}]
$$
这个公式描述了latent code的条件概率分布遵循Laplace分布的假设。
$\Pi_i$ 表示对所有空间位置i的乘积。
$C(\mu_{i,t},\sigma_{i,t}^2)$ 是Laplace分布的归一化常数, 依赖于位置i在时间t的均值 $\mu_{i,t}$ 和方差 $\sigma_{i,t}^2$。
$U(-\frac{1}{2})^{\hat{y}i,t}$ 是Laplace分布的未归一化部分。
$f_{hp}(f_{hp}(z_t))$、$f_{ar}(\hat{y}_{i,t})$ 和$f_{tp}(z_t)$ 提供了空间和时间的先验信息, 它们分别是hyper prior
解码器、auto regressive network和时间先验编码器的输出。

简单地说, 这个公式描述了在给定时间上下文 $z_t$ 的条件下, latent code $\hat{y}_t$ 的概率分布是如何
 根据Laplace分布来建模的。

**entropy model**

> 目前比较主流的深度学习图像压缩模型主要包含三个部分，编码网络（Encoder）、概率模型（Entropy Model）、解码网络（Decoder）。概率模型(Entropy Model)的作用是估计出压缩特征的熵概率模型，从而可以对压缩特征进行编解码。图像压缩中的熵概率模型一般以高斯分布作为先验，然后用模型去估计高斯分布的均值和方差，压缩特征基于该高斯分布得到的概率表用作编解码。概率模型估计得越准确，则消耗的码率越小。经过编解码后的重构 $\hat{y}$ 输入到Decoder中得到恢复的解压缩图像。现有的深度学习方法基本都是使用卷积神经网络，主要对局部相邻特征进行学习。但是在图像压缩任务中，还有一部分的全局冗余性没有被挖掘，这部分重复的特征会造成码率的重复消耗。所以，当前图像/视频压缩的关键改进点为设计更优秀的熵概率模型，在更少的压缩特征的码率消耗下提升输出图像的重建质量。

熵模型用于压缩量化隐编码 $\hat{y}_t$，HPE/HPD是超先验编码/解码器；AE/AD：算数编码/解码器。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221106111824239.png" alt="image-20221106111824239" style="zoom:40%;" />

​	熵模型中使用了3个先验：使用`hyper prior model`[Variational image compression with a scale hyperprior]来学习层次先验，使用`auto regressive network`[Joint autoregressive and hierarchical priors for learned image compression]来学习空间先验。这两种先验常用于图像压缩。latent codes也有时间上的关联性，因此设计了一个时间先验编码器来探索时间相关性，生成时间先验。结果是一个融合了空间和时间上下文信息的latent code，可以用于高效的编码和解码。



### Context Learning

​	特征 $\bar{x}_t$ 的学习仍采用运动补偿的思想，并且基于特征域而非像素域。设计上下文生成网络如下：
$$
\bar{x}_t = f_{context}(\hat{x}_{t-1}) = f_{cr}(warp(f_{fe}(\hat{x}_{t-1}),\hat{m}_t))
$$
​	首先通过一个特征提取网络 $\breve{x}_{t-1} = f_{fe}(\hat{x}_{t-1})$ 将参考帧从像素域转化至特征域；根据[Optical ﬂow estimation using a spatial pyramid network]中的光流估计学习上一重构帧 $\hat{x}_{t-1}$ 和当前帧 $x_t$ 之间的运动信息 $m_t$ ，之后对 $m_t$ 编解码；$\ddot{x}_t = warp(\breve{x}_{t-1},m_t)$ 得到初步的上下文 $\ddot{x}_t$；最后，通过上下文微调网络 $\bar{x}_t = f_{cr}(\ddot{x}_t)$ 获得最终的 $\bar{x}_t$ 。

```python
// input: referframe channel_num=3
// output: feature  channel_num=64 即将像素域转到高维特征域
self.feature_extract = nn.Sequential(
    nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
    ResBlock(out_channel_N, out_channel_N, 3),
)
```



### Training

$$
L = \lambda ·D+R
$$

D: MSE/MS-SSIM
R: latent code估计分布和真实分布的交叉熵

