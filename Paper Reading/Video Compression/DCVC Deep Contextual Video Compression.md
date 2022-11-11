# DCVC: Deep Contextual Video Compression



## Introduction

现有的神经视频压缩方法大多采用预测编码框架，首先生成预测帧，然后将其残差与当前帧进行编码。然而，对于压缩比，预测编码只是一种次优的解决方案，因为它使用简单的减法操作来去除帧间的冗余。在本文中，提出了一个深度上下文(deep contextual)视频压缩框架，以实现从预测编码到条件编码的转换。

从理论上讲，当前待编码的像素与之前所有已经重建的像素都可能有相关性。对于传统编码器，由于搜索空间巨大，使用人为制定的规则去显示地挖掘这些像素之间的相关性非常困难。因此残差编码假设当前像素只和预测帧对应位置的像素具有强相关性, 这是条件编码的一个特例。考虑到残差编码的简单性，最近的基于深度学习的视频压缩方法也大多采用残差编码，使用神经网络去替换传统编码器中的各个模块。

残差编码的信息熵 大于等于 条件编码的信息熵：$H(x_t - \bar{x_t}) \geq H(x_t|\bar{x_t})$；
已有的基于 auto-encoder的图像压缩，可通过autoencoder探索图像间的关系；本文提出构建基于条件编码的自编码器，代替残差编码。

## Related Work

现有的深度视频压缩工作可分为非时延约束和时延约束两类。

非时延约束：参考帧可以来自未来。[Video compression through image interpolation,]中，对之前的帧和未来帧进行帧插值得到预测帧，然后进行残差编码；[Neural inter-frame compression for video coding.]则引入光流运动估计。这种编码方式会带来更大的延迟，并且会显著增加GPU的内存成本。

时延约束：参考帧仅来自前一帧。DVC[DVC: an end-to-end deep video compression framework.]将传统视频编解码的模块均替换为network；[Improving deep video compression by resolutionadaptive ﬂow coding.]编码运动矢量时考虑 率失真优化。

## Proposed Method

### Framework of DCVC

**传统视频压缩 残差编码**
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t− \bar{x}_t) ⌉ ) + \bar{x}_t \ with \   \bar{x}_t= f_{predict}(\hat{x}_{t-1}).
$$

**条件编码**
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t|\bar{x}_t) ⌉ \ | \ \bar{x}_t )  \ with \   \bar{x}_t= f_{predict}(\hat{x}_{t-1}).
$$
然而，上述的条件编码仍然局限于低通道尺寸的像素域。所以，
$$
\hat{x}_t = f_{dec}(⌊f_{enc}(x_t|\bar{x}_t) ⌉ \ | \ \bar{x}_t )  \ with \   \bar{x}_t= f_{context}(\hat{x}_{t-1}).
$$
**Notations**

$x_t$: current frame
$\bar{x}_t$ : context.
$y_t$: latent code
$\hat{y}_t$: quantized $y_t$ (通过round操作)

对当前帧的encode和decode都基于 context值 $\bar{x}_t$，而 $x_t$ 与 $\bar{x}_t$ 的关系通过网络自适应地学习得到，而非以往通过相见操作得到残差 $r_t = x_t - \bar{x}_t$。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221101162222345.png" alt="image-20221101162222345" style="zoom:40%;" />

<center><strong>DCVC 结构图</strong></center>

### Entropy Model

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221106111824239.png" alt="image-20221106111824239" style="zoom:40%;" />

<center><strong>图 熵模型用于压缩 latent code.HPE/HPD: hyper prior encoder/decoder.AE/AD:arithmetic encoder/decoder.</strong></center>
