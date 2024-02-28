# A neural video codec with spatial rate-distortion control

## INTRO

用户无法灵活控制bit-rate和quality。

提出：全局R-D权衡参数、ROI掩码进行约束，实现了对ROI的bit-rate和质量的动态控制。实验结果：在复杂运动序列上表现最好，在ROI区域质量优于NON-ROI编解码器，Bjøntegaard-Delta率节省超过60%。



## RELATED WORK

### Variable bitrate compression

由于带宽的限制，实际需要部署多个单-bit rate模型来实现可变比特率，但造成了较大开销。多速率编解码器可在单个模型中支持一系列比特率[9,11,13,25,31,38,25]。

通过潜在缩放改变量化策略实现可变比特率。该方法通过在传输之前使用因子s缩放预量化潜在值来改变量化binwidth[9,13,25,31]。这种方法的主要优点是，理论上，单速率编解码器只需训练一个潜在扩展的辅助网络就可以变成可变速率编解码器。一个限制是缩放因子s必须与潜在z一起传输，但这种开销通常会随着传输数据的大小变化。

将R-D权衡参数β作为模型输入。在训练过程中，从预先指定的范围内采样不同的β参数，并相应地改变率失真损失。具体应用于图像[11,38]和视频[34,49]。本文参照此方法，利用β调节。



## METHOD

![image-20240227163545907](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20240227163545907.png)

### Loss Function

loss 函数目的是控制感兴趣区域 (ROI) 和非ROI区域之间的比特率分配。损失函数由几个部分组成，主要是速率失真权衡参数 $\beta$ 和一个给定的ROI掩码 $m$。这个损失函数用于在ROI和非ROI区域之间进行精确的质量控制。公式如下：
$$
C=E\begin{bmatrix}\beta_c\cdot R+\alpha\cdot C_{dist}^{ROI}+(1-\alpha)\cdot C_{dist}^{BG}\end{bmatrix}
$$
其中$C_{dist}^{ROI}$和$C_{dist}^{BG}$分别代表了ROI和背景的失真成本。这个方程确保了模型可以通过改变$\alpha$ 值来调整ROI和非ROI区域的编码质量。损失函数的设计使得网络能够在尽可能保持ROI质量的同时，压缩非ROl区域的数据。

**全局和局部码率控制**

global tradeoff parameter $\beta _c$；
local tradeoff parameter $\alpha_t$；
ROI mask：$m_t$；

