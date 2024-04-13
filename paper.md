# ROI-Aware Dynamic Network Quantization for Neural Video Compression



## Abstract

Deep neural networks have recently achieved great success in neural video compression(NVC), but due to the high complexity of deep video coding networks, NVC has not widely used in low-latency scenarios. Quantization is an effective way to reduce memory footprint and computational complexity for deep neural networks. However, existing methods overlook the unique characteristics of video frames and typically employ a fixed bit-width approach, which is suboptimal. In this paper, to achieve better frame reconstrction with lower computational complexity, we propose a ROI(Region of Interest)-aware dynamic quantization method for NVC networks that analyzes ROIs within video frames and dynamically alters the quantization strategy, allocating higher bit-widths for critical regions and lower bit-widths for less crucial areas. To this end, we present an efficient bit-allocator that adaptively determines quantization levels across different regions and frames, guided by the motion and texture complexity of the region. Experimental results conducted on 1080p standard test videos demonstrate effectiveness of the proposed dynamic quantization method.

## Introduction

 

With the rapid evolution of digital media technology, video has become a predominant internet data format. Efficient video encoding is vital for optimizing bandwidth and storage, particularly in online streaming, conferencing, and digital TV.

In the past decades, multiple generations of video coding standards have been developed, such as H.264/AVC \cite{wiegand2003overview}, H.265/HEVC \cite{sullivan2012overview}, and H.266/VVC \cite{bross2021overview}. These traditional coding standards are based on the predictive coding paradigm, and rely on handcrafted modules, such as block-based motion estimation and discrete cosine transform (DCT), to reduce the redundancies in video sequences. But the overall compression framework is not end-to-end optimized.

In recent years, neural video codec (NVC) has explored a completely new development direction. Existing deep video compression research can be categorized into two types according to the targeted scenarios: non-delay-constrained \cite{wu2018video,djelouah2019neural,yang2020learning,habibian2019video,pessoa2020end} and delay-constrained \cite{habibian2019video,lu2019dvc,lu2020end,hu2020improving,lin2020m,sheng2022temporal}. For the first type, the reference frame can come from positions after the current frame. This method involves significant latency, increased memory overhead, and higher computational demands due to multiple reference frames. For the low-latency type, reference frame comes from the previous reconstructed frame.

When designing video coding models for low-latency scenarios, a key challenge arises: these models are often required to be deployed on resource-constrained hardware devices, so model compression and acceleration become critical. However, existing video coding models have large memory and computational consumption.

To facilitate deployment, lots of works have been proposed. Quantization is one of the promising approaches for reducing the computational complexity of neural networks.



随着数字媒体技术的飞速发展，视频内容已成为互联网上最重要的数据形式之一。高效的视频编码对于带宽利用和数据存储具有重大意义，尤其在在线视频流、视频会议和数字电视等应用中尤为突出。

在过去几十年中，已经发展出多代视频编码标准，比如H.264/AVC、H.265/HEVC、H.266/VVC，这些传统编码标准都是基于预测编码范式，依靠手动设计的模块，例如基于块的运动估计和离散余弦变换（DCT）来减少视频序列中的冗余。尽管各模块都经过精心设计，但整个压缩系统并未进行端到端优化。

近年来，neural video codec已经探索了一个全新的发展方向。现有的深度视频压缩工作可分为无时延约束和时延约束两类。对于无时延约束编码，参考帧可以来自于当前帧之后的位置。由于存在多重参考帧，这种编码方式存在较大的延迟，且显著增加内存开销和计算消耗。对于低时延约束下的编码，参考帧来自于之前的重构帧。

在设计视频编码模型以应对低延迟应用场景时，我们面临一个关键挑战：这些模型常常需要在资源受限的移动设备上部署。为了适应这些设备的硬件限制，模型的压缩和加速变得至关重要，以确保能够高效地执行推理任务。然而，现有的视频编码模型往往对计算资源的需求极高，这在移动和低功耗设备上尤其成问题。

#TODO

因此，人们在网络压缩方面做了大量工作，来实现高效的推理。量化是降低神经网络计算复杂度的有效方法之一。我们发现，针对neura video codec的模型压缩工作很少，之前有人尝试对learned image compression进行量化。比如[^1][^2]中只对模型的权重进行量化，没有考虑对激活的量化；[^3][][][][][^4][][][][][^5] 同时考虑到了权重和激活值的量化，但他们均依赖于推理时的动态量化，根据图像的实时统计信息计算量化参数，然而这样需要很高的计算开销，不适用于低延迟场景下的深度视频编码。虽然以上方法为提高深度图像编码网络在低精度下的准确性和高效性做出了巨大努力，但并不完全适用于深度视频编码网络的特性：视频片段具有多样性，以同样的bit-width处理不同的视频片段限制了模型的性能；同时由于人眼对实时视频的不同区域关注度不同，对低关注度和高关注度区域采取相同的位宽无法有效分配计算资源，导致性能下降。因此我们设想，对更高关注度区域采取更高bit-width，低关注区域使用较低精度，可在保证不牺牲人眼关注区域精度的情况下有效实现计算节省。因此，本文旨在探索一种ROI区域感知的量化方法，实现更好的计算效率和重构质量的权衡。

To facilitate deployment, lots of works have been proposed to enable efficient inference. Quantization is one of the promising approaches for reducing the computational complexity of neural networks. We found little work on model compression for neura video codec and previous attempts to quantize learned image compression. For example, in [^1][^2], only the weights of the model are quantified, and the quantification of the activations is not considered. [^3][][][][][][^4][][][][][][][^5] consider both weight and activation quantization, but they all rely on dynamic quantization during inference, and calculate quantization parameters according to real-time image statistical information. However, this requires high computational overhead and is not suitable for depth video coding in low latency scenarios. Although the above methods have made great efforts to improve the accuracy and efficiency of deep image coding networks at low precision, they are not fully applicable to the characteristics of deep video coding networks: video clips are diverse, and processing different video clips with the same bit-width limits the performance of the model. At the same time, because the human eye pays different attention to different regions of real-time video, using the same bit width for low and high attention regions cannot effectively allocate computing resources, resulting in performance degradation. Therefore, we envisage that higher bit-width for higher attention regions and lower precision for low attention regions can effectively achieve computational savings without sacrificing the accuracy of the human eye attention regions. Therefore, this paper aims to explore an ROI region-aware quantization method that achieves a better trade-off between computational efficiency and reconstruction quality.



==#TODO==

本文提出一种简单的端到端网络量化方案，根据图x所示，根据输入视频片段的人眼关注度高/低区域，为不同区域动态选择量化位宽，同时兼顾高关注度区域的高质量重建和整体计算效率。具体而言，本文构建了一个轻量的bit allocator，用于决策每一帧的最佳bit分配，可以忽略其计算的开销。bit allocator实时决定每一帧中的不同区域采用什么bit-width，对高关注度区域分配更高的bit，反之对低关注度区域分配较低的bit。两个区域的特征分别传入单独进行编解码处理，在解码端完成融合并重构。对于网络权重，我们采用静态的量化方法，避免在编码端和解码端之间产生额外的量化信息传输。对于激活值，根据不同区域动态分配量化位宽，可以大大减少每个视频片段的计算量，实现效率和质量之间的更优权衡。

在几个1080p standard test videos（HEVC、UVG、MCL-JCV）上进行了测试，验证了本方法的有效性。实验结果表明，保持人眼关注区域的质量同时可达到 xxx 的码率节省，并且平均减少了xxx的flops和xxx的内存。

In this paper, we propose a simple end-to-end network quantization scheme, which dynamically selects  quantization bit-width for different regions according to the high/low human eye attention regions of the input video clip as shown in Figure x, while taking into account the high-quality reconstruction of the high attention regions and the overall computational efficiency. Specifically, a lightweight bit allocator is constructed to decide the best bit allocation for each frame, and the overhead of its computation can be ignored. The bit allocator determines in real time what bit-width to use for different regions in each frame, and assigns higher bits to high interest regions and lower bits to low interest regions. The features of the two regions were passed into the decoder for encoding and decoding separately, and the fusion and reconstruction were completed at the decoder. For the network weights, we adopt a static quantization method to avoid additional transmission of quantized information between the encoder and decoder. For the activation value, the dynamic allocation of quantization bit width according to different regions can greatly reduce the computation of each video segment and achieve a more optimal trade-off between efficiency and quality.

Experiments on several 1080p standard test videos (HEVC, UVG, MCL-JCV) verify the effectiveness of the proposed method. Experimental results show that xxx bitrate savings can be achieved while maintaining the quality of the region of interest, and xxx flops and xxx memory are reduced on average.

==#TODO==

In summary, the contributions of this work are:





## Related Works

### Network Quantization
Network quantization optimizes neural networks by converting 32-bit floating-point feature maps and weights into lower bit values, reducing computational complexity and memory usage \cite{krishnamoorthi2018quantizing,gholami2022survey,zhuang2021effective}.
For example, INQ \cite{zhou2017incremental}  introduce three interdependent operations, namely weight partition, group-wise quantization and re-training to optimize the training process.
DoReFa-Net \cite{zhou2016dorefa} exploits convolution kernels with low bit-width parameters and gradients to accelerate training and inference.
LQ-Net \cite{zhang2018lq} approximates the gradient propagation in quantization-aware training by straight-through estimator. Pact proposes a parameterized clipping activation function.
LSQ \cite{esser2019learned} introduces learnable quantization parameters and gradient-based optimization method, achieving commendable results in low-bit quantization.
AdaQuant proposed a post-training quantization method based on layerwise calibration and integer programming, which achieved ultra-low performance loss on 4-bit quantization.

Unlike traditional quantization methods that apply a single precision level across the entire neural network, mixed-precision quantization uses different precision levels for different parts of the network, which can effectively prevent significant performance degradation.
SDQ proposes to combine stochasticity with differentiable quantization to implement  mixed precision quantization.
Dong \etal proposed to decide the quantization bit-widths by Hessian's trace information.
Guo \etal proposes a single path one-shot model to address the weight co-adaption problem.

### Neural Video Compression

In recent years, the research of neural video codec technology have seen substantial advancements.
DVC is the first end-to-end optimized learned video compression method, which replaces each module in traditional video codec framework with convolutional neural networks.

==#TODO==

提一下 A neural video codec with spatial rate-distortion control这篇文章：采用预训练的

## Proposed Method

### Preliminaries

In a full-precision model, all weights and activations are represented using full-precision floating-point numbers (32 bits). Quantization refers to the process of converting these full-precision weights and/or activations into fixed-point numbers with a lower bit width, such as 2, 4, or 8 bits. Mixed-precision quantization involves representing different groups of neurons with varying quantization ranges (bit depths).

~~We build our model based on DCVC model and follow LSQ+ to build a quantization framework.~~ 

We build our model based on DCVC model to build a quantization framework.To ease the hardware implementation, we use ReLU as the nonlinearity after each layer of the network instead of a divisive normalization (GDN) block. Since GDN requires a extensive floating-point computations and has higher complexity. We use the straight-through estimator (STE) to preserve gradient information during the backward propagation phase. For all the convolutional weights and activations, we use channel-wise quantization parameters, since the range of weights and activations varies greatly between channels.

The process of for weights quantization is denoted as:
$$
\mathbf{w}_q=\left\lfloor\operatorname{clip}\left(\frac{\mathbf{w}}{s_{\mathbf{w}}},L_n^{\mathbf{w}},U_n^{\mathbf{w}}\right)\right\rceil,
$$


where $\mathbf{w}$ is full-precision weight, $\mathbf{w}_q$ is the quantized integer value, clip() clips the input with $L_{\mathbf{w}}$, $U_{\mathbf{w}}$,  $L_n^{\mathbf{w}}$, $U_n^{\mathbf{w}}$ are the integer lower and upper integer bounds of clip function respectively. n is the quantization bit-width. The quantization scale parameter, denoted as $s_w$​, determines the step size for the dequantized weight.
$$
\mathbf{x}_q=\left\lfloor\operatorname{clip}\left(\frac{\mathbf{x}}{s_{\mathbf{x}}},L_n^{\mathbf{x}},U_n^{\mathbf{x}}\right)\right\rceil
$$


The process of for activation quantization is denoted as:
$$
\boldsymbol{x}_q\equiv Q_b(\boldsymbol{x})=\lfloor\operatorname{clamp}(\boldsymbol{x},\alpha)\cdot\frac{s(b)}\alpha\rceil
$$

$$
\boldsymbol{\hat{x}} =\boldsymbol{x}_q · \frac{\alpha}{s(b)}
$$

$x_q$ is the quantized value of an real input value $x$, defined by the quantization function $Q_b(x)$. The function involves a clamping operation, denoted as clamp$( x, \alpha) $, which restricts the value of $x$ within the range $[-\alpha,\alpha]$,where $\alpha$ is a predefined or learnable upper bound. The result of the clamping operation is then scaled by the ratio of the step size $s(b)$ over $\alpha$. The step size $s(b)$ is typically a function of the bit-width $b$, where $s(b) = 2^{b-1}$. With $\lfloor·\rceil$ rounds the result down to the nearest integer value.Then the quantized value $x_q$ is converted back to its approximate real-value representation $\hat{x}$. To enable the optimization of the non-differentiable quantization process in an end-to-end manner，the gradient is derived by using the straight through estimator (STE) to approximate the gradient through the round function as a pass through operation.

Similarly, the process of for weights quantization is denoted as:
$$
\boldsymbol{w}_q\equiv Q_b(\boldsymbol{w})=\lfloor\operatorname{clamp}(\boldsymbol{w},\alpha)\cdot\frac{s(b)}\alpha\rceil
$$

$$
\boldsymbol{\hat{w}} =\boldsymbol{w}_q · \frac{\alpha}{s(b)}
$$



### Approach Overview

我们基于DCVC来构建视频编码量化网络。给定输入帧 $x_t$，目标是以尽可能少的比特代价重建高质量的视频帧 $\hat{x}_t$。首先估计输入帧$x_t$与前解码帧$\hat{x}_t$之间对应的运动信息 $v_t$。通过运动信息编码，得到解码后的运动 $\hat{v}_t$。上一解码的帧 $\hat{x}_{t-1}$ 经过特征提取后，与解码的运动 $\hat{v}_t$ 共同作为输入，得到高维context信息 $\bar{x}_t$ 。 $\bar{x}_t$用于当前帧 $x_t$ 的上下文编码和解码。最终得到重构的解码帧 $\hat{x}_t$。

图x显示了我们方法的概述。我们使用xxx来获取当前视频帧的显著性图，并生成人眼关注的显著区域和非显著区域的二值mask。在当前帧输入条件编码器之前，通过ROI mask图像将当前帧和预测光流的不同特征在编解码阶段分组处理。同时，我们制定了一个特定于图像帧的量化分配器，它实时为不同关注度的图像特征分配所需的精度，并在两组特征分组编解码过程中，不同区域的图像特征分别以被分配的精度完成训练和推理，且需要很少的计算成本。此外，由于视频片段中帧之间的连续性，当帧之间的内容变化不大时，当前帧的量化策略可沿用之前已传输帧的策略，以提高编解码效率。为了便于硬件实现，我们将GDN替换为ReLU，为了弥补替换GDN引起的非线性性缺失，我们引入了结构化知识转移来进一步提升模型的性能。



### Feature Dynamic Quantization

已有的工作通常没有考虑到视频片段的多样性，以及帧内的复杂性，大多数的量化bit-width分配方法通常是静态的，没有考虑到由于人眼对于帧内不同区域的关注度不同，导致所需的计算资源也是不同的。为了精确分配计算资源，我们提出动态bit-width分配，根据输入调整关键区域和非关键区域的bit-width。

假设ROI区域和Non-ROI区域均有K个不同量化bit-width候选 $b^1,b^2,...,b^K$​​​，bit-allocator将根据量化strategy为不同区域特征分别分配一个最优的量化bit-width。每一个量化bit-width $b^k$ 对应的特征量化函数为：
$$
Q_{b^k}(\boldsymbol{x})=\lfloor\operatorname{clamp}(\boldsymbol{x},\alpha_k)\cdot\frac{s(b_k)}{\alpha_k}\rceil
$$
其中 $\alpha^k$表示第k个bit-width对应的scale parameter， $s(b^k) = 2^{b^k-1}$ 表示integer range of $b^k$ from a set of $K$​ bit-width options. 为了实现动态的帧内区域量化，我们通过bit-allocator为每一个bit-width分配一个概率，因此ROI/Non-ROI区域的动态量化可写作：
$$
\boldsymbol{x}_{q}=\sum^KQ_{b^k}(\boldsymbol{x})\cdot P^{k}(\boldsymbol{x})\\
s.t. \sum_{k=1}^KP^k(\boldsymbol{x})=1.
$$
其中 $P^k(\boldsymbol{x})$ 表示分配给bit-width $b^k$ 的概率。



### Bit-allocator for Dynamic Quantization

为了进行bit-width的选择，我们使用一个轻量的bit-allocator，通过判断图像区域特征的多种复杂性来预测其对应的bit-width，具体来说，分别为ROI区域和Non-ROI区域配置两个bit-allocator，ROI区域相比Non-ROI区域具有更高精度的bit-width候选。bit-allocator将输出由每个bit-width候选的选择概率组成的概率向量。



![截屏2024-04-08 00.06.27](/Users/liujiamin/Library/Application Support/typora-user-images/截屏2024-04-08 00.06.27.png)

==#TODO:==

Bit-allocator由xxx组成，经由softmax进一步计算bit-width的预测概率：
$$
\pi(\boldsymbol{x}) = \mathrm{Softmax}{(fc(\sigma(\boldsymbol{x}),|\nabla I|))} \\
\sum_K \pi^k(x) = 1
$$
bit-allocator的输出logits为 $\pi^1,\pi^2,...,\pi^k$，对于某一输入，根据argmax选定分配概率最大的量化bit-width：
$$
\boldsymbol{x}_{q} =Q_{b^k}(\boldsymbol{x})=\arg\max_{Q_{b^k}(\boldsymbol{x})}\pi^k(\boldsymbol{x})
$$
但是从离散分布 $\pi(x)$ 的采样过程是不可微的，为了使优化问题完全可微，采用Gumbel-Softmax方法来提供一个采样argmax的可微公式：
$$
P^k(\boldsymbol{x})=\frac{\exp\left((\log\pi^k(\boldsymbol{x})+g^k)/\tau\right)}{\sum_{j\in\Omega}\exp\left((\log\pi^j(\boldsymbol{x})+g^j)/\tau\right)},
$$
$P^k(x)$ denotes the soft assignment probability of bit-width $b^k$, $g^k$ is a random noise drawn from a Gumbel(0, 1) distribution. $\tau$ is the temperature parameter which controls the discreteness of the output distribution. As $\tau$ approaches 0, the output approximates a hard max function, resulting in a vector close to onehot encoding. As $\tau$ increases, the distribution becomes more uniform, meaning the probability is more evenly spread across the candidates. Following the setting of [], during training phase, we initialize  parameter τ at 5 and progressively decrease it to 0 through an annealing process .



### Loss and Training Process

本文采用DCVC中的视频编码框架来证明我们所提方法的有效性。视频压缩通过率失真优化在编码过程中平衡压缩比和视频质量，以最小化编码后的视频码率和编码引起的失真。整体的编码框架通过优化以下率失真权衡来实现：
$$
L_t=\lambda D_t+R_t=\lambda d(x_t,\hat{x}_t)+[H(\hat{y}_t)+H(\hat{m}_t)]
$$
$L_t$ is the total loss function for current time step $t$, which comprises two parts: distortion term $D_t$ and bit rates term $R_t$. $d(x_t, \hat{x}_t)$ measures the distortion between original frame $x_t$ and reconstructed frame $\hat{x}_t$. Distortion can be quantified by MSE(mean squared error) and  MS-SSIM (multiscale structural similarity). $\lambda$ is a Lagrange multiplier that controls the trade-off between distortion $D_t$ versus rate $R_t$. $H(\hat{y_t})$ and $H(\hat{m}_t)$ represents the bit rates of latent representations $\hat{y}_t$ and $\hat{m}_t$.



| Stage   | Training Modules |                  Loss                  | Steps | LR        | Frames |
| ------- | ---------------- | :------------------------------------: | ----- | --------- | ------ |
| Stage 1 | MV-E/D           |          $\lambda D_{warp}+R$          | 300k  | $1E^{-4}$ | 2      |
| Stage 2 | MV-E/D, ME       |          $\lambda D_{warp}+R$          | 200k  | $1E^{-4}$ | 2      |
| Stage 3 | Contextual-E/D   |         $\lambda D_{recon}+R$          | 300k  | $1E^{-4}$ | 2      |
| Stage 4 | All              |         $\lambda D_{recon}+R$          | 400k  | $1E^{-4}$ | 2      |
| Stage 5 | All              | $\frac{1}{T}\sum^T{(\lambda D_t+R^t)}$ | 800k  | $1E^{-4}$ | 5      |

我们使用渐进式训练策略训练整个编码框架，分为5个阶段。首先，我们固定其他网络模块参数，只训练motion vector(MV) encoder(E)/decoder(D)模块；之后，联合训练MV-E/D和motion estimation(ME)；之后，固定MV-E/D和MV网络参数，训练Contextual E/D模块；最后，开放网络所有参数，整个网络联合以端到端方式训练。

在前两阶段，loss函数由参考帧 $\hat{x}_{t-1}$ 经过重构mv的warp后引起的失真与mv的bit rates构成。
$$
L_t = \lambda D_{warp} + R_{mv} = \lambda d(x_t, \ddot{x}_t) + H(\hat{m}_t)
$$

> loss including the distortion of the reference frame warped by the compressed motion and the bit-rate for compressing mv。

第3阶段之后，MV的估计和编解码网络已收敛，我们固定这一部分参数，对条件编解码的部分参数进行训练。loss函数由重构帧和原始帧之间的失真以及所有latent code的bit rates组成。
$$
L_t = \lambda D_{recon} + R = \lambda d(x_t, \hat{x}_t) + [H(\hat{m}_t)+ H(\hat{y}_t)]
$$

前4阶段只采用两个连续时间步的帧进行训练，忽略了 $\hat{x}_t$ 对于下一帧 $x_{t+1}$ 的潜在误差影响，从而导致导致误差传播。因此在最后一个阶段，将 $\hat{x}_t$ 作为 $x_{t+1}$ 编码阶段的参考帧，获得多个连续时间间隔的损失，其中T表示时间间隔，实验中我们设置为5.
$$
L^T=\frac{1}{T}\sum_{t=1}^TL_t=\frac{1}{T}\sum_{t=1}^T\{\lambda d(x_t,\hat{x}_t)+[H(\hat{y}_t)+H(\hat{m}_t)]\}
$$




#### Weight Quantization

采用固定bit-width





### 结构性知识转移

衡量编码后两个分布的差异？KL散度？不知道咋办啊啊啊啊啊啊







[^1]: H. Sun, L. Yu, and J. Katto, “Learned image compression with fixed-point arithmetic,” in Proc. Picture Coding Symposium, 2021.
[^2]: End-to-end learned image compression with fixed point weight quantization.



[^3]:W. Hong, T. Chen, M. Lu, S. Pu, and Z. Ma, “Efficient neural image decoding via fixed-point inference,” IEEE Trans. Circuits and Systems for Video Technology, vol. 31, no. 9, pp. 3618–3630, 2020.
[^4]: H. Sun, L. Yu, and J. Katto, “Q-lic: Quantizing learned image compression with channel splitting,” IEEE Trans. Circuits and Systems for Video Technology, 2022 (early access).
[^5]: End-to-end learned image compression with quantized weights and activations



## Experiments



### Experiments Setup

**Datasets** training dataset是Vimeo-90K septuplet dataset，测试数据包括HEVC类B (1080P)， C (480P)， D (240P)， E (720P)来自通用测试条件，由编解码器标准社区使用。此外，还对MCL-JCV和UVG数据集的1080p视频进行了测试。

**Implementation Details.** 我们的深度编码模型基于DCVC条件编码模型框架，并基于UNISAL模型生成显著性mask图，以此得到帧中的ROI区域和Non-ROI区域。受到xx的启发，我们采用渐进式训练框架，分为5个阶段进行训练。

For the learning rate, it is set as 1e-4 at the start and 1e-5 at the fine-tuning stage. The training batch size is set as 4. For comparing DCVC with other methods, we follow [5] and train 4 models with different λ s {MSE: 256, 512, 1024, 2048; MS-SSIM: 8, 16, 32, 64}.

The learning rate is initially set as 10−4 for all loss functions (1), (2), (3) and (4). When training the whole network by the final loss of (4), the learning rate decreases by the factor of 10 after convergence until 10−6.

In OpenDVC, we first follow DVC [9] to train the PSNR- optimized model with the distortion D as the Mean Square Error (MSE) and λ = 256, 512, 1024 and 2048. Then, the MS-SSIM models are fine-tuned only using the final loss function (4) with D = 1 − MS-SSIM. The MS-SSIM mod- els with λ = 8, 16, 32 and 64 are fine-tuned from the pre- trained PSNR models with λ = 256, 512, 1024 and 2048, respectively. Note that, we use BPG [4] to compress the I-frames for the PSNR models in OpenDVC, and use the learned image compression method [7] to compress the I- frames for the MS-SSIM models. Specifically, the BPG with QP = 37, 32, 27 and 22 is used for PSNR models with λ = 256, 512, 1024 and 2048, respectively. The MS-SSIM models with λ = 8, 16, 32 and 64 use [7] with the quality levels of 2, 3, 5 and 7, respectively.

Datasets In the training stage, we use the Vimeo-90k dataset[39]. Vimeo-90k is a widely used dataset for low-level vision tasks [35,20]. It is also used in the recent learning based video compression tasks [19,14]. To evaluate the compression performance of different methods, we employ the following datasets,

HEVC Common Test Sequences [28] are the most popular test sequences for evaluating the video compression performance [28]. The contents in common test sequences are diversified and challenging. We use Class B(1920 × 1080), Class C(832 × 480) and Class D(352 × 288) in our experiments.

Video Trace Library(VTL) dataset [3] contains lots of raw YUV se- quences used for the low-level computer vision tasks. Following the setting in [14], we use 20 video sequences with the resolution of 352 × 288 in our experi- ments, and the maximum length of the video clips is set to 300 for all sequences.

Ultra Video Group(UVG) dataset [2] is a high frame rate(120fps) video dataset, in which the motion between neighbouring frames is small. Following the setting in [38,19,16], we use the video sequences with the resolution of 1920×1080 in our experiments.

MCL-JVC dataset [34] consists of 24 videos with the resolution of 1920 × 1080. This dataset is widely used for video quality assessment. For a fair comparison with [14], we also include this dataset in our experiments.

Implementation details. We train four models with different λ values (256, 512, 1024, 2048) in Eq. (1). To generate the I-frame/key-frame for video com- pression, we use the learning based image compression method in [9], in which the corresponding λ in the image codec are empirically set to 1024, 2048, 4096 and 12000, respectively.

In our implementation, we use DVC [19] as the baseline method. In the training stage, the whole network is first optimized by using the loss in Eq.(1), then is fine-tuned based on the error propagation aware loss in Eq.(2). The corresponding batch sizes are set to 4 and 1, respectively. The resolution of the training images is 256×256. We use Adam optimizer [17] and the initial learning rate is set as 1e−4. In the inference stage, the encoder is also optimized by using

**Bit-width candidates**



### Ablation study



### Qualitative Results



### 



Dynamic Network Quantization for Efficient Video Inference

参与决策的条件：光流复杂度、结构/纹理复杂度



要不要蒸馏呢？有可能不需要

实验结果怎么证明：列举：

QP = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}

1. 是采用多种量化方法？还是一种量化方法作为基线 ：表格/R-D图
   1. 多种量化方法：需要对比：静态下的psnr、bpp/动态下的psnr（稍差）、bpp（节省）、ROI/Non-ROI区域分别的psnr、bpp（对比）
   2. 一种量化方法作基线：全精度、静态下的psnr、bpp/动态下的psnr（稍差）、bpp（节省）、ROI/Non-ROI区域分别的psnr、bpp（对比）比多种方法多一个全精度
2. 图片结果：
3. 不同帧的不同区域的bit-width选择结果
4. 定性结果





## Conclusion

