# DVC: An End-to-end Deep Video Compression



## Introduction 

- 传统视频压缩算法依靠手动设计的模块，例如基于块的运动估计和离散余弦变换（DCT）来减少视频序列中的冗余。尽管各模块都经过精心设计，但整个压缩系统并未进行端到端优化。 期望通过联合优化整个压缩系统来进一步提高视频压缩性能。

- 使用基于光流的运动估计，与传统基于块的运动估计相比，可能增加存储运动信息所需的bits数。

- 率失真优化

​	提出end-to-end网络模型，其中视频压缩的关键模块（运动估计、运动补偿、变换、量化、熵编码）都通过该模型实现，且和传统的视频压缩算法模块有一对一的关系。

## Related Work

### Image Compression

- 基于RNN的图像压缩；
- 基于CNN设计自编码器风格的网络；
- 广义除法归一化（generalized divisive normalization）（GDN）、多尺度图像分解（multi-scale image decomposition）、对抗训练（adversarial training）、重要性图（importance map）和帧内预测（intra prediction），现有的以上工作为本文网络的重要组成部分。

### Video Compression 

- 大部分仍沿用传统的视频压缩算法，手工设计，无法以端到端的方式联合优化；
- 已有的部分基于DNN的帧内预测、残差编码针对一特定模块，非端到端的压缩方式；
- `Video compression through image interpolation. In ECCV, September 2018.`基于RNN的压缩方法，采用帧插值(frame interpolation), 但仍采用基于块的运动估计；同时只考虑了原始帧和重构帧之间的失真(MSE)最小化，未考虑整个编解码过程中**率失真**的平衡。



## Proposed Method

​	**Notations**
​	${x_1,x_2,...,x_{t-1},x_{t},...}$ 表示视频帧序列，$x_t$表示 $t$ 时间步时刻的视频帧；
​	$v_t$: 块运动向量/光流信息；
​	$\hat{v}_t$: 重构运动信息；
​	$\bar{x}_t$: 预测帧；
​	$\hat{x}_t$: 重构帧；
​	$r_t$: 原始帧 $x_t$和预测帧 $\bar{x}_t$之间的残差 $r_t = x_t - \bar{x}_t$；
​	$\hat{r}_t$: 重构/解码残差；
​	对残差值$r_t$和运动向量$v_t$进行线性/非线性转换后，进行量化，使用如下公式中的符号进行表示：
$$
r_t \stackrel{transform}{\longrightarrow} y_t \stackrel{quantize}{\longrightarrow} \hat{y}_t \stackrel{decoder}{\longrightarrow} \hat{r}_t\\
v_t \stackrel{transform}{\longrightarrow} m_t \stackrel{quantize}{\longrightarrow} \hat{m}_t \stackrel{decoder}{\longrightarrow} \hat{v}_t
$$

<img src="https://img-blog.csdnimg.cn/20201010163822194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x3cGx3Zg==,size_16,color_FFFFFF,t_70#pic_center" style="zoom: 70%;" />



### Classic Framework of Video Compression

​	传统的视频压缩框架将输入帧分成相同大小的 $n\times n$ 块，并基于块进行运动估计。Figure 2 中，编码器端包含所有模块，解码器端不包含蓝色模块。压缩框架中编码器端的过程如下。

​	**Step 1.  运动估计**
​	估计当前帧 $x_t$和前一重构帧 $\hat{x}_{t-1}$ 之间的运动信息，得到每个块对应的运动矢量 $v_t$。

​	**Setp 2. 运动补偿**
​	基于 **Step 1**中定义的运动矢量 $v_t$，通过将前一重构帧中的对应像素复制到当前帧来获得预测帧 $\bar{x}_t$ 。原始帧 $x_t$ 与预测帧 $\bar{x}_t$之间的残差为 $r_t = x_t - \bar{x}_t$ 。

​	**Step 3. 变换与量化**
​	**Step 2**的残差 $r_t$量化为 $\hat{y}_t$. 量化前使用线性变换（如DCT，变换至频域获得更好的压缩效果）。

​	**Step 4. 逆变换**
​	**Step 3**中的量化结果 $\hat{y}_t$ 通过逆变换获得重构残差 $\hat{r_t}$ 。

​	**Step 5. 熵编码**
​	**Step 1**中的运动矢量 $v_t$ 和 **Step 3**中的残差量化结果 $\hat{y}_t$ 熵编码后发送到解码器端。

​	**Step 6. 帧重构**
​	**Step 2**中的预测帧 $\bar{x}_t$ 和 **Step 4**中的重构残差 $\hat{r}_t$相加，$\hat{x}_t = \bar{x}_t + \hat{r}_t$ ,获得重构帧。$\hat{x}_t$ 将被用于第 $t+1$ 步的运动估计。

​	解码器端：根据发送到解码器端的 $v_t$ 和 $\hat{y}_t$ ，在Step 2中进行运动补偿，得到 $\bar{x}_t$，并在  **Step 4** 中对 $\hat{y_t}$ 反量化得到 $\hat{r_t}$，然后在**Step 6** 中进行帧重构获得 $\hat{x_t}$ .

### Proposed Mehtod

​	相较于传统压缩模型，采用CNN估计光流获取运动信息 $v_t$ , 并采用MV编解码器来压缩编码光流值。两种算法框架差异和关系如下：

​	**Step N1.  运动估计与压缩**
​	采用CNN[Optical Flow Estimation using a Spatial Pyramid Network]估计光流值 $v_t$；设计一个MV自编码器[End-to-end optimized image compression. 2016]压缩运动信息 $v_t$。
$$
v_t \stackrel{encoder}{\longrightarrow} m_t \stackrel{quantization}{\longrightarrow} \hat{m}_t \stackrel{decoder}{\longrightarrow} \hat{v}_t
$$
​	**Setp N2. 运动补偿**
​	基于 **Step N1**中获得的光流 $\hat{v}_t$ ，通过运动补偿获得预测帧 $\bar{x}_t$ 。原始帧 $x_t$ 与预测帧 $\bar{x}_t$之间的残差为 $r_t = x_t - \bar{x}_t$ 。

​	**Step N3-N4. 变换，量化与反变换**
​	使用非线性的变换替代了 **Step 3**中的线性变换。

​	**Step N5. 熵编码**
​	**Step N1**中的量化运动信息 $\hat{m}_t$ 和 **Step N3**中的残差量化结果 $\hat{y}_t$ 熵编码后发送到解码器端。

​	**Step N6. 帧重构**
​	**Step N2**中的预测帧 $\bar{x}_t$ 和 **Step N4**中的重构残差 $\hat{r}_t$相加，$\hat{x}_t = \bar{x}_t + \hat{r}_t$ ,获得重构帧。$\hat{x}_t$ 将被用于第 $t+1$ 步的运动估计。



### Motion Estimation

参考 Optical Flow Estimation using a Spatial Pyramid Network.

在DVC基础上改进：pyramid估计光流和整个压缩系统联合优化。下图右侧为联合优化重建光流，在人体等较为平滑的区域像素零值更多，更易于压缩。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/20221031100621.png" style="zoom:50%;" />



### MV Encoder and Decoder Network

​	MV编解码器采用了一种自编码风格的网络。

<img src="https://img-blog.csdnimg.cn/20201010171408351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x3cGx3Zg==,size_16,color_FFFFFF,t_70#pic_center" style="zoom:66%;" />

**Auto-Encoder Network [End-to-end optimized image compression. 2016]**

​	作者在这篇文章中提出了一种端到端的图像压缩框架，它由analysis transform、均匀量化器和synthesis transform组成。它的analysis transform过程包含三个重复的阶段，每个阶段包括卷积线性滤波器以及非线性激活函数；在这里，联合非线性用来实现局部增益控制。通过将原来不可导的量化函数替代为连续的proxy function，就可以采用SGD在训练集上联合的优化整个模型的率失真性能。在特定的情况下，松弛的损失函数可以看做通过VAE生成模型的对数似然。实验证明，在PSNR或者MS-SSIM的测量标准下，端到端的方法显著优于JPEG以及JPEG-2000算法。

​	为了压缩图像，需要对图像进行量化，然而量化同时也会带来图形的失真，所以需要在图像的大小（离散熵）和量化误差（失真）之间做平衡。

​	由于在高维空间中对图像做量化困难，所以有损压缩通常采用变换编码形式，将图像转换到合适的空间做处理，如图所示：

<img src="https://img-blog.csdnimg.cn/202012042010161.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70" style="zoom:67%;" />

<center>Fig. 非线性变换编码框架</center>

​	$\mathbf{x}$与 $\mathbf{\hat{x}}$ 分别表示输入的原图和经过编解码器后的重建图片。$g_a$表示编码器提供的非线性变换 $\mathbf{y} = g_a(\mathbf{x},\phi)$，$\mathbf{y}$量化得到 $q$ 后，反量化/重建得到 $\hat{y}$ ，再通过 $g_s$ 解码器重建图片结果。
​	估计得到 $R$ (码率) ,计算原图$\mathbf{x}$与 $\mathbf{\hat{x}}$ 的失真得到 $D$，D的度量指标如PSNR,MS-SSIM，或者其他感知域如 VMAF 等。对$R$ 和 $D$ 进行**率失真联合优化**，定义损失函数为：
$$
L = \lambda \ · D + R
$$
**为什么使用GDN**

1. 常见的压缩方法通常采用正交线性变换去除数据的相关性从而简化熵编码，然而线性滤波器响应的联合统计特性表现出很强的高阶依赖，他们可以被联合局部非线性增益控制操作所降低。此外，通过在线性块变换编码中引入局部归一化可以提高编码性能，也能提高CNN的目标检测性能。

2. 在高斯化图像的局部联合统计特性时，GDN比（线性变换和逐点非线性的级联）效果要好。



### Motion Compensation Network

​	给定前一重构帧 $\hat{x}_{t-1}$ 和运动矢量 $\hat{v}_t$，运动补偿网络获得预测帧 $\hat{x}_t$。为了消除伪影，将变换帧 $warp(\hat{x}_{t-1},\hat{v}_t)$，参考帧 $\hat{x}_{t-1}$ 和运动矢量 $\hat{v}_t$ wrap操作输入一个CNN中。基于像素级的运动补偿可以有效避免传统的基于块的运动补偿方法导致的块效应。

<img src="https://img-blog.csdnimg.cn/2020101017145968.png#pic_center" style="zoom:67%;" />

​	**运动补偿CNN结构** 

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221024095332005.png" alt="image-20221024095332005" style="zoom:30%;" />

### Residual Encoder and Decoder Network

参考 Variational image compression with a scale hyperprior.

将残差信息 $r_t$ 通过非线性网络(Variational image compression with a scale hyperprior) 进行编码/压缩。



### Training Strategy

**损失函数**
$$
\lambda D + R = \lambda d(x_t, \hat{x}_t) + (H(\hat{m}_t) + H(\hat{y}_t))
$$
​	$d(x_t, \hat{x}_t)$ 表示 $x_t, \hat{x}_t$ 之间的失真，采用 MSE评估。
​	$H(·)$ 表示用于编码的比特数。
​	$\lambda$ 是拉格朗日乘数，决定number of bits 和 distortion之间的权衡。

**量化** 

​	量化后会导致梯度几乎处处为0，无法训练。这里通过在训练阶段**加入均匀噪声来代替量化运算**。以 $y_t$ 为例，训练阶段的量化表示 $\hat{y}_t$ 是通过在 $y_t$ 上加上均匀噪声 $\eta$ 来近似得到的，即 $\hat{y}_t = y_t + \eta$ ；在推理阶段，四舍五入取整，即 $\hat{y}_t = round(y_t)$ .

**比特率估计**

参考 Variational image compression with a scale hyperprior.



## Experiments

### Setup

**评估指标：** PSNR，MS-SSIM

**Detail：** 使用不同 $\lambda$ (256, 512, 1024, 2048) , adam优化器，初始学习率0.0001，损失趋于稳定时，学习率除以10. 训练图像 256 $\times$ 256 .

### Results

**H.264：**以 PSNR 和 MS-SSIM 为测度时，DVC性能优于H.264；
**H.265：**以 MS-SSIM 为测度时，DVC优于H.265。

