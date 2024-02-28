# Exploring Sparsity in Image Super-Resolution for Efficient Inference



本文提出了一种用于图像超分辨率（Super-Resolution, SR）的新型卷积神经网络（CNN）结构——稀疏掩码超分辨率网络（Sparse Mask Super-Resolution, SMSR）。此方法旨在通过学习和应用**空间和通道掩码**来提高图像超分辨率的推理效率，尤其是在移动设备上。

## Introduction

- 现有的CNN基础的超分辨率方法通常对所有位置使用相同的计算资源，这样做在计算效率低下。
- 图像的低分辨率版本中的大部分细节通常存在于边缘和纹理区域，而在较平坦的区域则需要较少的计算资源。
- 论文中提出的SMSR网络通过学习空间掩码来修剪冗余计算，空间掩码识别“重要”的区域，而通道掩码则用于标记这些“不重要”区域中的冗余通道。
- 通过Gumbel softmax技巧使得二值掩码在训练中可微分，在推理中则使用稀疏解决方案来跳过冗余计算。
- SMSR网络展示了在减少计算复杂度的同时，能够保持或提升图像超分辨率的性能。



图2，给一张高清图片HR image $I^{HR}$和一张低视效的图$I^{LR}$,我们对$I^{LR}$进行Bicubic和RCAN的超分操作获得$I^{SR}_{Bicubic}$和$$I^{SR}_{RCAN}$$。图二中显示了$I^{SR}_{RCAN},I_{Bicubic}^{SR}$和$I^{HR}$在亮度信道上的绝对区别。从图二可以观察到$I_{Bicubic}^{SR}$对于单调的区域是“足够好的”。只有一小部分区域的细节明显缺失。即SR任务在空间域上具有本质上的稀疏性。与Bicubic相比，RCAN在边缘区域的性能更好 (-17%的像素|$I$|>0.1) ,而在平坦区域的性能也相当(图2(c))。虽然RCAN的重点是恢复边缘区域的高频细节(图2(d)), 但这些平坦区域同时也同样被处理。因此，涉及到冗余计算。

 图3给出了RCAN骨干块中ReLU层之后的特征映射。可以看出，不同通道的空间稀疏性存在显著差异。此外，相当多的通道非常稀疏(稀疏度0.8)，只有边缘和纹理区域被激活。也就是说，这些平坦区域的计算是冗余的，因为这些区域在ReLU层之后没有被激活。总之，RCAN只激活不重要区域(如平坦区域)的少量通道，而激活重要区域(如边缘区域)的更多通道。

![image-20231225181833835](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231225181833835.png)

## Method

 在这些观察的激励下，我们学习稀疏掩模来定位并跳过冗余计算来进行有效的推理。具体来说，我们的空间掩码动态地识别重要区域，而通道掩码则标记那些不重要区域中的冗余通道。与网络剪枝方法[10,30,14]相比，我们考虑了区域冗余，只剪枝不重要区域的通道。与自适应推理网络[37,27]不同，我们进一步研究通道维上的冗余，在更细粒度的水平上本地化冗余计算。

![](https://img-blog.csdnimg.cn/20210914154610705.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAbTBfMzc4NjAwNzY=,size_20,color_FFFFFF,t_70,g_se,x_16)

###  Sparse Mask Generation

![image-20231225181856530](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231225181856530.png)

**spatial mask 空间掩码**

空间掩码的目的是为了识别特征图中的重要区域。**为了让二值掩码 (0代表“不重要”, 1代表“重要”) 可学习，论文采用了Gumbel softmax分布来近似one-hot分布**。输入的特征首先通过一个沙漏形网络块产生$F^{spa}$,然后使用Gumbel softmax技巧得到一个软化的空间掩码$M^{spa}$。

在方程(1)中，$x,y$是垂直和水平索引，$G^{spa}$是一个Gumbel噪声张量，其所有元素遵循 Gumbel(0,1)分布，$\tau$是一个温度参数。随着$\tau$趋于无穷大，Gumbel softmax分布中的样本将变得均匀。当$\tau$趋于0时，样本将变成one-hot分布。在实践中，训练开始时$\tau$较高，随后逐渐降低至较低的温度以生成二值空间掩码。
$$
M_k^{spa}[x,y]=\frac{\exp\Bigl(\Bigl(F^{spa}[1,x,y]+G_k^{spa}[1,x,y]\Bigr)/\tau\Bigr)}{\sum_{i=1}^2\exp\Bigl(\Bigl(F^{spa}[i,x,y]+G_k^{spa}[i,x,y]\Bigr)/\tau\Bigr)},
$$
**channel mask 通道掩码**
$$
M_{k,l}^{ch}[c]=\frac{\exp\Bigl(\bigl(S_{k,l}[1,c]+G_{k,l}^{ch}[1,c]\bigr)/\tau\Bigr)}{\sum_{i=1}^2\exp\Bigl(\bigl(S_{k,l}[i,c]+G_{k,l}^{ch}[i,c]\bigr)/\tau\Bigr)},
$$
通道掩码的目的是为了识别冗余通道 (0代表“冗余”, 1代表“保留”)。这里，$c$是通道索引，$S_{k,l}$是第$k$个稀疏掩码模块 (SMM) 中第$l$层卷积层的辅助参数， $G_k^{ch}$是Gumbel噪声张量。

**sparsity regularization 稀疏性正则化**
$$
\eta_{k,l}=\frac{1}{C\times H\times W}\sum_{c,x,y}\begin{pmatrix}(1-M_{k,l}^{ch}[c])\times M_{k}^{spa}[x,y]\\+M_{k,l}^{ch}[c]\times I[x,y]\end{pmatrix},
$$

$$
L_{reg} = \frac{1}{K\times L}\sum_{k,l}\eta_{k,l}
$$

其中，$\eta_{k,l}$是稀疏项，表示在特征图中激活位置的比率，$L_{reg}$是稀疏正则化损失，用于鼓励网络产生更稀疏的特征，$F$是一个全1的张量，$K$ 是SMM的数量，$L$是每个SMM中稀疏掩码卷积层的数量。



### Training Strategy

1. 训练阶段为了在所有位置允许梯度的反向传播，训练阶段并不执行真正的稀疏卷积。相反，它通过将标准“密集”卷积的结果与掩码相乘，得到“稀疏”特征图。具体来说，输入特征 $F$ 与 $M_k^{ch}$ 和 $1-M_k^{ch}$ 相乘得到 $F^D$和$F^S$，即“密集”和“稀疏”特征图，然后 $F^D$和$F^S$ 分别通过共享权重的两个卷积层。输出特征图与掩码 $1-M_k^{ch}$、$M_k^{ch}$ 和 $M_k^{spa}$ 的组合相乘来激活特征图的不同部分，最后，所有features 相加以生成最终的输出特征$F^{out}$。

2. 推理阶段

   在推理阶段，基于预测的空间和通道掩码，执行真正的稀疏卷积。例如，在第$k$个SMM的第$l$层，卷积核首先被分割成四个子核，然后$F^D$和$F^S$分别通过这四个子核的卷积生成四个不同的结果。值得注意的是，这些操作生成了“密集”的卷积$F^D2D$和$F^D2S$,以及“稀疏”的卷积$F^S2D$和$F^S2S$。通过稀疏掩码卷积，能够更有效率地计算特征，并且从这些特征中获得的信息用于重建最终的超分辨率图像。

**本质也是将feature mask获得不同部分后concatenate，再得到一个新的融合特征**
