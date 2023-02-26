# DCVC: Deep Contextual Video Compression



## 简介

​	在本文中，提出了一个深度上下文(deep contextual)视频压缩框架，以实现从预测编码到条件编码的范式转换。
​	从理论上讲，当前待编码的像素与之前所有已经重建的像素都可能有相关性。对于传统编码器，由于搜索空间巨大，使用人为制定的规则去显示地挖掘这些像素之间的相关性非常困难。因此残差编码假设当前像素只和预测帧对应位置的像素具有强相关性, 这是条件编码的一个特例。考虑到残差编码的简单性，最近的基于深度学习的视频压缩方法也大多采用残差编码，使用神经网络去替换传统编码器中的各个模块。
​	可知，残差编码的信息熵 大于等于 条件编码的信息熵：$H(x_t - \bar{x_t}) \geq H(x_t|\bar{x_t})$；
​	已有的基于 auto-encoder的图像压缩，可通过 autoencoder 探索图像间的关系；本文提出构建基于条件编码的自编码器，代替残差编码。

## 相关工作

​	现有的深度视频压缩工作可分为非时延约束和时延约束两类。
​	非时延约束：参考帧可以来自未来。[Video compression through image interpolation,]中，对之前的帧和未来帧进行帧插值得到预测帧，然后进行残差编码；[Neural inter-frame compression for video coding.]则引入光流运动估计。这种编码方式会带来更大的延迟，并且会显著增加GPU的内存成本。
​	时延约束：参考帧仅来自前一帧。DVC[DVC: an end-to-end deep video compression framework.]将传统视频编解码的模块均替换为network；[Improving deep video compression by resolutionadaptive ﬂow coding.]编码运动矢量时考虑 率失真优化。

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

对当前帧的encode和decode都基于 context值 $\bar{x}_t$，而 $x_t$ 与 $\bar{x}_t$ 的关系通过网络自适应地学习得到，而非以往通过相减操作得到残差 $r_t = x_t - \bar{x}_t$。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221101162222345.png" alt="image-20221101162222345" style="zoom:40%;" />




对 latent code $y_t$ 进行量化和算数编码操作，运动信息 $m_t$ 通过MV Encoder压缩后均传送至解码器端。



### Entropy Model

​	熵模型用于压缩量化隐编码 $\hat{y}_t$，HPE/HPD是超先验编码/解码器；AE/AD：算数编码/解码器。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221106111824239.png" alt="image-20221106111824239" style="zoom:40%;" />

​	熵模型中使用了3个先验：层次先验[Variational image compression with a scale hyperprior]，空间先验[Joint autoregressive and hierarchical priors for learned image compression]，和时间先验。



### Context Learning

​	特征 $\bar{x}_t$ 的学习仍采用运动补偿的思想，并且基于特征域而非像素域。设计上下文生成网络如下：
$$
\bar{x}_t = f_{context}(\hat{x}_{t-1}) = f_{cr}(warp(f_{fe}(\hat{x}_{t-1}),\hat{m}_t))
$$
​	首先通过一个特征提取网络 $\breve{x}_{t-1} = f_{fe}(\hat{x}_{t-1})$ 将参考帧从像素域转化至特征域；根据[Optical ﬂow estimation using a spatial pyramid network]中的光流估计学习上一重构帧 $\hat{x}_{t-1}$ 和当前帧 $x_t$ 之间的运动信息 $m_t$ ，之后对 $m_t$ 编解码；$\ddot{x}_t = warp(\breve{x}_{t-1},m_t)$ 得到初步的上下文 $\ddot{x}_t$；最后，通过上下文微调网络 $\bar{x}_t = f_{cr}(\ddot{x}_t)$ 获得最终的 $\bar{x}_t$ 。

### Training

$$
L = \lambda ·D+R
$$

D: MSE/MS-SSIM
R: latent code估计分布和真实分布的交叉熵
