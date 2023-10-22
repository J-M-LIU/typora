# Temporal Context Mining for Learned Video Compression



## Introduction

现有深度学习视频编码方向：

1. **基于残差编码的方案**：首先在像素域或特征域应用预测，然后压缩预测的残差。
2. **基于体积编码的方案**：视频被视为包含多个帧的体积。应用3D卷积以获取潜在表示。
3. **基于熵编码的方案**：每一帧都使用独立的图像编解码器进行编码。探索不同帧中的潜在表示之间的相关性，以指导熵建模。
4. **基于条件编码的方案**：时态上下文作为条件，编码器自动探索时态相关性。

对比DCVC：DCVC从先前解码的帧中生成单尺度时态上下文。但是，先前解码的帧丢失了很多纹理和运动信息，因为它只包含三个通道。此外，单尺度上下文可能不包含足够的时态信息。因此，本文试图找到更好的方法来学习和利用时态上下文。

**贡献**：

- 提出了一个**时态上下文挖掘（TCM）模块**，以学习更丰富、更准确的时态上下文。考虑到学习单尺度上下文可能无法很好地描述时空非均匀性，TCM模块采用了分层结构来生成多尺度时态上下文。
- 提出了时态上下文重新填充（TCR）的方法；
- 相比于DCVC，本文的entropy model中不使用auto regressive 熵模型（自回归熵模型可以提高压缩比，但会大大增加解码时间，不利于并行化）使用`hyper prior model`[Variational image compression with a scale hyperprior]学习层次先验，并结合temporal prior。

## Methodology

![image-20231007220706592](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231007220706592.png)

### Temporal Context Mining

之前的条件编码方案（DCVC）：从上一个解码帧$\hat{x}_{t-1}$中提取context，但由于$\hat{x}_{t-1}$只包含3个通道，丢失了很多信息，因此本文提出从$\hat{x}_{t-1}$生成过程中的最后一个卷积层的特征 $F_{t-1}$ 中来获取。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231009190112491.png" alt="image-20231009190112491" style="zoom:50%;" />

1. 使用具有$L$ 个级别的特征提取模块从传播的特征$F_{t-1}$生成多尺度特征$F_{t-1}^l$。该模块由卷积层和残差块组成。

$$
  F_{t-1}^l=extract(F_{t-1}),l=0,1,2
$$

2. 解码的MV $\hat{v}_t$ 使用双线性滤波器进行down sample, 生成多尺度MV $\hat{v}_t^l$, 其中 $\hat{v}_t^0$ 被设置为 $\hat{v}_t$ 。
    注意, 每个下采样的MV都被除以2。然后, 对MV $\hat{v}_t^l$ 和多尺度特征$F_{t-1}$ 进行warp操作。

$$
\bar{F}_{t-1}^l=warp(F_{t-1}^l,\hat{v}_t^l),l=0,1,2
$$
3. 使用由一个subpixel层和一个redisual block 组成的上采样模块upsample $\bar{F}_{t-1}^{l+1}$。然后, 将上采样的特征与相同尺度的$\bar{F}_{t-1}^l$进行拼接。
   $$
   \tilde{F}_{t-1}^l=concat(\bar{F}_{t-1}^l,upsample(\bar{F}_{t-1}^{l+1})),l=0,1
   $$

4. 在分层结构的每个级别,使用由一个卷积层和一个残差块组成的上下文细化模块学习残
    差。残差被添加到$\bar{F}_{t-1}^l$以生成最终的时间上下文$\bar{C}_t^l$。

$$
\bar{C}_t^l=\bar{F}_{t-1}^l+refine(\tilde{F}_{t-1}^l),l=0,1,2
$$

### Temporal Context Re-ﬁlling

**为什么要重新填充时间上下文？**在视频中，连续的帧往往具有很高的时间相关性。通过将这些时间上下文信息重新填充到编码/解码过程中，可以更好地利用这种相关性，从而实现更高效的视频压缩。（类比DCVC中，相似的步骤是将 $x_t$ 和 context $\bar{x}_t$ 一同送入contextual encoder中。）

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231010121111260.png" alt="image-20231010121111260" style="zoom:50%;" />

- Contextual Encoder-Decoder and Frame Generator：拼接最大scale的temporal context $\bar{C}_t^0$ 和 $x_t$，一起传入contextual encoder；拼接 $\bar{C}_t^1$ 和 $\bar{C}_t^2$ 然后也传入encoder；decoder的过程与之相逆，详见 Fig. 5. 
- **bottleneck residual block**：由于concatenation会导致通道数增加，数据维度变大，因此构建了bottleneck残差块来降低复杂度。
