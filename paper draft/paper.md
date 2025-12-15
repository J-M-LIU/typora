# ROI-Aware Dynamic Network Quantization for Neural Video Compression



## Abstract

Deep neural networks have recently achieved great success in neural video compression(NVC), but due to the high complexity of deep video coding networks, NVC has not widely used in low-latency scenarios. Quantization is an effective way to reduce memory footprint and computational complexity for deep neural networks. However, existing methods overlook the unique characteristics of video frames and typically employ a fixed bit-width approach, which is suboptimal. In this paper, to achieve better frame reconstrction with lower computational complexity, we propose a ROI(Region of Interest)-aware dynamic quantization method for NVC networks that analyzes ROIs within video frames and dynamically alters the quantization strategy, allocating higher bit-widths for critical regions and lower bit-widths for less crucial areas. To this end, we present an efficient bit-allocator that adaptively determines quantization levels across different regions and frames, guided by the motion and texture complexity of the region. Experimental results conducted on standard test videos demonstrate effectiveness of the proposed dynamic quantization method.

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

因此，人们在网络压缩方面做了大量工作，来实现高效的推理。量化是降低神经网络计算复杂度的有效方法之一。我们发现，针对neura video codec的模型压缩工作很少，之前有人尝试对learned image compression进行量化。比如[^1][^2]中只对模型的权重进行量化，没有考虑对激活的量化；[^3][][][][][^4][][][][][^5] 同时考虑到了权重和激活值的量化，但他们均依赖于推理时的动态量化，根据图像的实时统计信息计算量化参数，然而这样需要很高的计算开销，不适用于低延迟场景下的深度视频编码。虽然以上方法为提高深度图像编码网络在低精度下的准确性和高效性做出了巨大努力，但并不完全适用于深度视频编码网络的特性：视频片段具有多样性，以同样的bit-width处理不同的视频片段限制了模型的性能；同时由于人眼对实时视频的不同区域关注度不同，对低关注度和高关注度区域采取相同的位宽无法有效分配计算资源，导致性能下降。因此我们设想，对更高关注度区域采取更高bit-width，低关注区域使用较低精度，可在保证不牺牲人眼关注区域图像质量的情况下有效实现计算节省。因此，本文旨在探索一种ROI区域感知的量化方法，实现更好的计算效率和重构质量的权衡。

To facilitate deployment, lots of works have been proposed to enable efficient inference. Quantization is one of the promising approaches for reducing the computational complexity of neural networks. We found little work on model compression for neura video codec and previous attempts to quantize learned image compression. For example, in [^1][^2], only the weights of the model are quantified, and the quantification of the activations is not considered. [^3][][][][][][^4][][][][][][][^5] consider both weight and activation quantization, but they all rely on dynamic quantization during inference, and calculate quantization parameters according to real-time image statistical information. However, this requires high computational overhead and is not suitable for depth video coding in low latency scenarios. Although the above methods have made great efforts to improve the accuracy and efficiency of deep image coding networks at low precision, they are not fully applicable to the characteristics of deep video coding networks: video clips are diverse, and processing different video clips with the same bit-width limits the performance of the model. At the same time, because the human eye pays different attention to different regions of real-time video, using the same bit width for low and high attention regions cannot effectively allocate computing resources, resulting in performance degradation. Therefore, we envisage that higher bit-width for higher attention regions and lower precision for low attention regions can effectively achieve computational savings without sacrificing the accuracy of the human eye attention regions. Therefore, this paper aims to explore an ROI region-aware quantization method that achieves a better trade-off between computational efficiency and reconstruction quality.



==#TODO==

本文提出一种简单的端到端网络量化方案，根据图x所示，根据输入视频片段的人眼关注度高/低区域，为不同区域动态选择量化位宽，兼顾高关注度区域的高质量重建和整体计算效率。具体而言，本文构建了一个轻量的bit allocator，用于决策每一帧的最佳bit分配，可以忽略其计算的开销。bit allocator实时决定每一帧中的不同区域采用什么bit-width，对高关注度区域分配更高的bit，反之对低关注度区域分配较低的bit。两个区域的特征分别传入单独进行编解码处理，在解码端完成融合并重构。对于网络权重，我们采用静态的量化方法，避免在编码端和解码端之间产生额外的量化信息传输。对于激活值，根据不同区域动态分配量化位宽，可以大大减少每个视频片段的计算量，实现效率和质量之间的更优权衡。

在几个 standard test videos（HEVC、UVG、MCL-JCV）上进行了测试，验证了本方法的有效性。实验结果表明，保持人眼关注区域的质量同时可达到 xxx 的码率节省，并且平均减少了xxx的flops和xxx的内存。

In this paper, we propose a simple end-to-end network quantization scheme, which dynamically selects  quantization bit-width for different regions according to the high/low human eye attention regions of the input video clip as shown in Figure x, while taking into account the high-quality reconstruction of the high attention regions and the overall computational efficiency. Specifically, a lightweight bit allocator is constructed to decide the best bit allocation for each frame, and the overhead of its computation can be ignored. The bit allocator determines in real time what bit-width to use for different regions in each frame, and assigns higher bits to high interest regions and lower bits to low interest regions. The features of the two regions were passed into the decoder for encoding and decoding separately, and the fusion and reconstruction were completed at the decoder. For the network weights, we adopt static quantization method to avoid additional transmission of quantized information between the encoder and decoder. For the activation value, the dynamic allocation of quantization bit width according to different regions can greatly reduce the computation of each video segment and achieve a more optimal trade-off between efficiency and quality.

In this paper, we propose a simple end-to-end network quantization scheme that dynamically selects quantization bit-widths for ROI/non-ROI regions of an input video clip based on eye fixation saliency, as shown in Figure X. This approach not only maintains high-quality reconstruction in areas of high attention but also considers overall computational efficiency. Specifically, we use a lightweight bit allocator that decides the optimal bit allocation strategy for each frame, with negligible computational overhead. Bit allocator assigns higher bit-width to areas of high saliency and lower bit-width to areas of low saliency. Features from these two regions are separately encoded and decoded, and then fused and reconstructed at the decoder. For the weights, we use fixed bit-width quantization to avoid additional quantization information transfer in bitstream. For activations, dynamically allocating quantization bit-widths based on different regions significantly reduces the computational, achieving a better trade-off between efficiency and quality.

Experiments on standard test videos (HEVC, UVG, MCL-JCV) verify the effectiveness of the proposed method. Experimental results show that xxx bitrate savings can be achieved while maintaining the quality of the region of interest, and xxx flops and xxx memory are reduced on average.





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

尽管最近取得了进展，但视频识别模型的量化问题很少被探索。此外，现有方法以固定计算成本的静态方式执行量化，使以输入为条件的自适应量化成为开放问题。

尽管已有的工作取得了great success，但是针对learned video compression的模型量化还未被探索。此外，现有的量化方法没有考虑到视频帧内容的多样性，以固定计算资源的方式静态地量化。因此，如何自适应分配计算资源来实现高效量化成为关键问题。

Despite the great success of existing works, model quantization for learned video compression has not been explored. In addition, the existing quantization methods do not consider the diversity of video frame content and quantize statically with fixed computing resources. Therefore, how to adaptively allocate computing resources to achieve efficient quantization becomes a key issue.

### Neural Video Compression

In recent years, the research of neural video codec technology have seen great advancements. Among them, neural video compression in low-latency scenarios has received extensive attention. DVC is the first end-to-end optimized learned video compression method, which replaces each module in traditional video codec framework with convolutional neural networks. 2.improves efficiency by combining optical flow and flow prediction in a learning-based approach. 3. introduces a simplified video compression method that improves motion handling by introducing a scale parameter to optical flow. 4.uses multiple resolutions to better compress motion data, optimizing the process both for entire video frames and individual blocks.



1. Dvc: An end-to-end deep video compression framework.

2. Learned Video Compression via Joint Spatial-Temporal Correlation Exploration.
3. Scale-space flow for end-to-end optimized video compression.
4. Improving deep video compression by resolution-adaptive flow coding



在delay-constrained 的场景下的工作取得了很多进展，除此之外，一些工作研究了learned compression的高效架构和加速推理[1,2,3]。

Significant progress has been made in delay-constrained scenarios, and additionally, some studies have explored efficient architectures and accelerated inference for learned compression network. 4. using implicit neural representations, enhanced by  quantization and meta-learning to speed up encoding.  5. enhances neural data compression by fully adapting the entire autoencoder model to specific videos. 6. map coordinates to pixel values, modulating inputs for efficient motion compensation, and reducing bitrate with learned integer quantization. However, most methods do not consider the deployment on resource-limited devices, and how to achieve efficient inference on mobile and low-power devices remains a challenge.

1. Computationally Efficient Neural Image Compression

2. Conditional entropy coding for efficient video compression
3. Real-Time Adaptive Image Compression.
4. Implicit Neural Representations for Image Compression
5. Overfitting for Fun and Profit: Instance-Adaptive Data Compression
6. Implicit Neural Video Compression





Despite the progress in developing neural image and video compression solutions, literature that explicitly aims to improve their efficiency and practical usefulness is limited. Efficient architectures are explored in [15, 18, 33, 34]. Overfitting to a video or segments of a video is another means to reduce model complexity while improving rate-distortion performance [40, 44, 49]. Yet, in their current form, these methods are unable to run in real-time on resource-constrained mobile devices. To the best of our knowledge, these exists only a single public neural video codec deployed to a mobile device [32] that runs in all-intra mode, i.e., all frames are coded as images.

尽管在开发神经图像和视频压缩解决方案方面取得了进展，但明确旨在提高其效率和实际用途的文献仍然有限。[15, 18, 33, 34]探索了高效的架构。对视频或视频片段的过拟合是降低模型复杂度同时提高率失真性能的另一种方法[40,44,49]。然而，以目前的形式，这些方法无法在资源受限的移动设备上实时运行。据我们所知，只有一个部署到移动设备[32]上的公共神经视频编解码器，以全帧内模式运行，即所有帧都编码为图像。





Methods in the deep residual coding category [4, 10, 12, 14, 15, 21, 25, 28, 45]) follow traditional video compression frameworks. They perform predictive coding (e.g., motion compensation) and encode residual information. The pioneering work DVC [28] replaces all key coding operations with CNNs in the traditional residual cod- ing pipeline, enabling end-to-end optimization. Most subsequent works build upon this pipeline and improve performance using more powerful modules and advanced techniques. For example, to produce better-aligned context (e.g., frame or feature), M-LVC [21] uses a multi-frame alignment strategy, while FVC [16] adopts a deformable convolutional warping technique.

The works in the deep contextual coding category [13, 17, 18, 24, 36] extend the generative-based NIC methods and build spatio- temporal conditional entropy models using spatial and temporal contexts. Lombardo et al. [24] produced the dynamic global and local latent variables, while Habibian et al. [13] adopted a 3D-based VAE with a gated mechanism to generate temporal context. Dif- ferent from the aforementioned methods using all accumulated information, Li et al. proposed DCVC [17], which directly adopts a motion compensation strategy to generate temporal context from the adjacent compressed frame. To enhance the temporal informa- tion, Sheng et al. [36] further improved DCVC by using multi-scale temporal context mining. The most recent version of DCVC (re- ferred to as DCVC* in this work) with hybrid entropy models [18] outperforms the traditional video coding standard H.266/VVC [8].

Nevertheless, to the best of our knowledge, most NVC research focuses on producing better spatial and temporal information in- dividually rather than on how to aggregate and leverage them effectively. For instance, deep residual coding methods simply sub- tract the temporal information, whereas deep contextual coding ap- proaches mostly adopt common operations (such as concatenation) to combine the learned temporal and spatial contexts. Instead of such simple combinations, we propose a transformer-based module ST-XCT, which leverages the powerful cross-covariance attention mechanism to support better exploitation of the spatio-temporal correlation.

提一下 A neural video codec with spatial rate-distortion control 这篇文章：采用预训练的语义分割模型，对于在真实视频场景中更为多样的实例，可能面临语义不充足和缺失的情况，不利于实例内容丰富的场景。因此我们考虑一种针对人眼关注点感知的编码模型。

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


The process for activation quantization is denoted as:
$$
\boldsymbol{x}_q\equiv Q_b(\boldsymbol{x})=\lfloor\operatorname{clamp}(\boldsymbol{x},\alpha)\cdot\frac{s(b)}\alpha\rceil
$$

$$
\boldsymbol{\hat{x}} =\boldsymbol{x}_q · \frac{\alpha}{s(b)}
$$

$x_q$ is the quantized value of an input feature $x$, defined by the quantization function $Q_b(x)$. The function involves a clamping operation, denoted as $clamp( x, \alpha) $, which restricts the value of $x$ within the range $[-\alpha,\alpha]$,where $\alpha$ is a predefined or learnable upper bound. The result of the clamping operation is then scaled by the ratio of the step size $s(b)$ over $\alpha$. The step size $s(b)$ is typically a function of the bit-width $b$, where $s(b) = 2^{b-1}$. With $\lfloor·\rceil$ rounds the result down to the nearest integer value.Then the quantized value $x_q$ is converted back to its approximate real-value representation $\hat{x}$. To enable the optimization of the non-differentiable quantization process in an end-to-end manner，the gradient is derived by using the straight through estimator (STE) to approximate the gradient through the round function as a pass through operation.

Similarly, the process of for weights quantization is denoted as:
$$
\boldsymbol{w}_q\equiv Q_b(\boldsymbol{w})=\lfloor\operatorname{clamp}(\boldsymbol{w},\alpha)\cdot\frac{s(b)}\alpha\rceil
$$

$$
\boldsymbol{\hat{w}} =\boldsymbol{w}_q · \frac{\alpha}{s(b)}
$$

**修改**

The quantization function $Q_b(x)$ quantizes an input feature $x$ to its quantized version $x_q$. The clipping operation denoted as $\operatorname{clip}(z, q_1, q_2) $  constrains input $z$ with boundaries  $q_1$ and $q_2$. $\frac{\alpha_x}{s(b_x)}$ is the quantizer step size;  $s(b_x)$ is typically calculated as $2^{b_x-1}$; $\alpha_x$ is a learnable scale parameter  as mentioned in [LSQ]. And $Q_N^{b_x} = 2^{b_x-1}$, $Q_P^{b_x} = 2^{b_x-1}-1$. The $\lfloor·\rceil$ rounds result to the nearest integer. To enable the optimization of the non-differentiable quantization process in an end-to-end manner，the gradient is derived by STE to approximate the gradient through the round function as a pass through operation. Similarly, weight quantization process can be described as:
$$
\boldsymbol{x}_q = Q_b(\boldsymbol{x})=\lfloor\operatorname{clip}(\boldsymbol{x}\cdot\frac{s(b_x)}{\alpha_x}, -Q_N^{{b_x}}, Q_P^{{b_x}})\rceil \cdot \frac{\alpha_x}{s(b_x)},
$$


$$
\boldsymbol{w}_q\ = Q_b(\boldsymbol{w})=\lfloor\operatorname{clip}(\boldsymbol{w}\cdot\frac{s(b_w)}{\alpha_w}, -Q_N^{{b_w}}, Q_P^{{b_w}})\rceil · \frac{\alpha_w}{s(b_w)}
$$



### Approach Overview

我们基于DCVC 来构建视频编码量化网络。给定输入帧 $x_t$，视频编码的目标是以尽可能少的比特代价重建高质量的视频帧 $\hat{x}_t$。首先估计输入帧$x_t$与前解码帧$\hat{x}_t$之间的运动信息 $v_t$。经过运动信息编解码网络，得到解码后的运动 $\hat{v}_t$。对上一解码的帧 $\hat{x}_{t-1}$ 进行特征提取，然后与解码的运动 $\hat{g}_t$ 共同作为输入来获取高维context信息 $\bar{x}_t$ 。 $\bar{x}_t$用于当前帧 $x_t$ 的上下文编码和解码。最终得到重构的解码帧 $\hat{x}_t$。

图x显示了我们方法的概述。我们使用UNISAL来获取当前视频帧的显著性图，并生成人眼关注的显著区域和非显著区域的二值mask $m_t$​。首先，通过 $m_t$ 划分当前帧和预测光流的ROI和non-ROI特征，concat后始终进行分组卷积。同时，我们制定了一个特定于图像帧的bit-width allocator，它基于区域的图像结构的复杂度和运动复杂度，为不同显著性的区域分配optimal的精度，并在两组特征分组编解码过程中，不同区域的图像特征分别以被分配的精度完成训练和推理。此外，由于视频片段中帧之间的连续性，当帧之间的内容变化较小时，当前帧的量化策略可沿用之前已传输帧的策略，以提高编解码效率。为了便于硬件实现，我们将GDN替换为ReLU，为了弥补替换GDN引起的非线性性缺少，我们通过知识蒸馏来进一步提升模型的性能。



我们通过 mask $m_t$ 划分出 current frame /predicted optical flow的 ROI和non-ROI

We build our video compression quantization baseline based on DCVC. Given an input frame $x_t$, the target of video codec is to reconstruct a high-quality video frame $\hat{x}_t$ with least bitrate. The motion information $v_t$ between input frame $x_t$ and the previous decoded frame $\hat{x}_t$ is first estimated. After motion encoder and decoder, the decoded motion $\hat{v}_t$ is obtained. Feature extraction is performed on last decoded frame $\hat{x}_{t-1}$, and then combined with the decoded motion $\hat{g}_t$ as input to get high-dimensional context information $\bar{x}_t$. $\bar{x}_t$ is used for contextual encoding and decoding of the current frame $x_t$. Finally, the reconstructed decoded frame $\hat{x}_t$ is obtained.

 

Figure x provides an overview of our approach. We use xxx to obtain the saliency map of the current video frame and generate a binary mask $m_t$​ of the salient and non-salient regions, corresponding to ROI and non-ROI regions. Before the current frame is input into the conditional encoder, the different features of the current frame and the predicted optical flow are grouped and processed in the encoding and decoding stage through the ROI mask image. At the same time, we develop a frame-specific bit-width allocator, which assigns the required accuracy to the image features of different attention levels in real time based on the complexity of image structure and motion of the region. In the process of encoding and decoding of the two groups of features, the image features of different regions are trained and inferred with the assigned accuracy respectively. And it requires little computational cost. In addition, due to the continuity between frames in a video clip, when the content between frames does not change much, the quantization strategy of the current frame can follow the strategy of the previously transmitted frame to improve the coding and decoding efficiency. To facilitate hardware implementation, we replace GDN with ReLU, and to compensate for the lack of nonlinearity caused by replacing GDN, we further improve the performance of the model by knowledge distillation.



First,  the ROI and non-ROI features of the current frame and the predicted optical flow are segmented using $m_t$, and then concatenated for group convolution. Then, we use frame-specific bit-width allocator that assigns optimal bit-width to regions of varying saliency based on the complexity of image structure and motion information. During the grouped encoding and decoding processes, image features and motion info of ROI/non-ROI regions are processed at assigned bit-widths, for both training and inference. Moreover, due to the continuity between frames in a video clip, when there is minimal change between frames, the quantization strategy of the current frame can adopt the strategy of previously transmitted frames to enhance encoding and decoding efficiency. To facilitate hardware implementation, we replaced GDN with ReLU. To compensate for the loss of nonlinearity caused by replacing GDN, we further enhance model performance through knowledge distillation.



Additionally, due to the continuity between frames in a video segment, when there is minimal change in content from one frame to another, the quantization strategy of the current frame can leverage the strategy of previously transmitted frames to improve encoding and decoding efficiency. To facilitate hardware implementation, we replaced GDN with ReLU. To compensate for the loss of nonlinearity resulting from the replacement of GDN, we further enhance the model's performance through knowledge distillation.

 droste2020unified



### Feature Dynamic Quantization

已有的工作通常没有考虑到视频片段的多样性，以及帧内的复杂性，大多数的量化bit-width分配方法通常是静态的，没有考虑到由于人眼对于帧内不同区域的关注度不同，如果对ROI区域保留更多计算资源，并且适当降低non-ROI区域的重建精度，可以很好地提升模型效果。为了精确分配计算资源，我们提出动态bit-width分配，根据输入调整关键区域和非关键区域的bit-width。

Existing works do not consider the diversity of video clips and the intra-frame complexity. Most of the quantized bit-width allocation methods are usually static, and do not take into account that the human eye pays different attention to different regions in the frame, if  we reserve more computing resources for ROI region and appropriately reduce the reconstruction accuracy in non-ROI regions, the model effect can be well improved. To accurately allocate computing resources, we propose dynamic bit-width allocation to adjust the bit-width of salient and less-salient regions according to the input.

Existing work usually does not take into account the diversity of video clips and intra-frame complexity. Most of the quantized bit-width allocation methods are usually static, and do not take into account that due to the different attention of human eyes to different regions in the frame, if more computing resources are reserved for ROI regions and the reconstruction accuracy of non-ROI regions is appropriately reduced, the reconstruction accuracy of non-ROI regions is appropriately reduced. Can improve the performance of the model very well. To accurately allocate computing resources, we propose dynamic bit-width allocation to adjust the bit-width of critical and non-critical regions according to the input.

假设ROI区域和Non-ROI区域均有K个不同量化bit-width候选 $b^1,b^2,...,b^K$，bit-allocator将根据量化strategy为输入的区域特征 $\boldsymbol{x}$ 分别分配一个最优的量化bit-width。每一个量化bit-width $b^k$ 对应的特征量化函数为：
$$
Q_{b^k}(\boldsymbol{x})=\lfloor\operatorname{clamp}(\boldsymbol{x},\alpha^k)\cdot\frac{s(b^k)}{\alpha^k}\rceil
$$
其中 $\alpha^k$表示第k个bit-width对应的scale parameter， $s(b^k) = 2^{b^k-1}$ 表示integer range of $b^k$ from a set of $K$​ bit-width options. 为了实现动态的帧内区域量化，我们通过bit-allocator为每一个bit-width分配一个概率，因此ROI/Non-ROI区域的动态量化可写作：
$$
\boldsymbol{x}_{q}=\sum^KQ_{b^k}(\boldsymbol{x})\cdot P^{k}(\boldsymbol{x})\\
s.t. \sum_{k=1}^KP^k(\boldsymbol{x})=1.
$$
其中 $P^k(\boldsymbol{x})$ 表示分配给bit-width $b^k$ 的概率。



### Bit-allocator for Dynamic Quantization

为了进行bit-width的选择，我们使用一个轻量的bit-allocator，根据图像区域特征的复杂度来分配bit-width，具体来说，分别为ROI区域和Non-ROI区域配置两个bit-allocator，ROI区域相比Non-ROI区域具有更高精度的bit-width候选。受到[xxx]的启发，我们以帧内的结构复杂度和运动复杂度分别为两个区域制定量化策略。对于 $t$ 时刻的输入帧区域特征 $x^R_t$，位宽分配器将输出由每个位宽候选项的概率 $\pi^k(\boldsymbol{x})$​ 组成的概率向量：

To select the bit-width, we use a lightweight bit-allocator that assigns bit-width based on complexity of region features. Specifically, we configure two bit-allocators for ROI region and Non-ROI region, respectively. ROI regions have higher bit-width candidates than non-ROI regions. Inspired by xxx, we formulate quantization strategies for the two regions in terms of structure complexity and motion complexity within the frame, respectively. Bit-allocators output a probability vector consisting of probability $\pi^k(\boldsymbol{x})$ for each bit-width candidate:
$$
\pi(\boldsymbol{x}) = \mathrm{Softmax}{\Big(fc\big(\sigma(\boldsymbol{x}),G(\boldsymbol{x}),\sigma(\boldsymbol{v}),G(\boldsymbol{v})\big)\Big)}
$$
其中 $\sigma(\boldsymbol{x})$ 和 $\sigma(\boldsymbol{v})$ 分别表示区域特征和预测光流的标准差，$G(\boldsymbol{x})$ 和 $G(\boldsymbol{v})$ 表示区域特征和光流向量在水平和垂直方向上的梯度。这几个指标评估了图像的纹理结构复杂度和运动复杂度，复杂结构或具有高运动内容的区域将分配更高bit-width。 我们将这几个指标concat后输入全连接层 $fc$，经由Softmax获得bit-allocator的输出logits $\pi^1,\pi^2,...,\pi^k$​，对应K个bit-width candidates。

Where $\sigma(\boldsymbol{x})$ and $\sigma(\boldsymbol{v})$ represent  the standard deviation of region features and predicted motion information, respectively. $G(\boldsymbol{x})$ and $G(\boldsymbol{v})$ represent the gradients of region features and optical flow vectors in horizontal and vertical directions. These several metrics evaluate the texture structure complexity and motion complexity of the image, and regions with complex structures or high motion content will be assigned higher bit-width. We concat these indicators into the fully connected layer $fc$, and obtain the output logits $\pi^1,\pi^2,... ,\pi^k$, corresponding to K bit-width candidates.

对于某一输入，通过argmax选定分配概率最大的量化bit-width：
$$
\boldsymbol{x}_{q} =Q_{b^k}(\boldsymbol{x})=\arg\max_{Q_{b^k}(\boldsymbol{x})}\pi^k(\boldsymbol{x})
$$
但是从离散分布 $\pi(x)$​ 的采样过程是不可微的，为了使优化问题完全可微，采用Gumbel-Softmax方法来提供一个采样argmax的可微公式：

But the sampling process from the discrete distribution $\pi(x)$ is not differentiable. To make the optimization problem fully differentiable, we use Gumbel-Softmax Samlping to provide a differentiable formula for the sampled argmax:
$$
P^k(\boldsymbol{x})=\frac{\exp\left((\log\pi^k(\boldsymbol{x})+g^k)/\tau\right)}{\sum_{j\in\Omega}\exp\left((\log\pi^j(\boldsymbol{x})+g^j)/\tau\right)},
$$
$P^k(x)$ denotes the soft assignment probability of bit-width $b^k$, $g^k$ is a random noise drawn from a Gumbel(0, 1) distribution. $\tau$ is the temperature parameter which controls the discreteness of the output distribution. As $\tau$ approaches 0, the output approximates a hard max function, resulting in a vector close to onehot encoding. As $\tau$ increases, the distribution becomes more uniform, meaning the probability is more evenly spread across the candidates. Following the setting of [], during training phase, we initialize  parameter τ at 5 and progressively decrease it to 0 through an annealing process .



### Loss and Training Process

视频压缩通过率失真优化平衡压缩比和视频质量，以最小化编码后的视频码率和失真。Similar to xxx, 我们使用一个控制ROI和Non-ROI区域重建质量的因子 $\beta$，给定ROI mask $m_t$​，整体的编码框架通过优化以下率失真权衡来实现：

Video compression uses rate-distortion optimization to balance compression ratio and video quality in the encoding process to minimize the encoded video bitrate and the distortion caused by coding. Similar to [xxx], using a factor $\beta$ that controls the reconstruction quality of ROI and Non-ROI regions, given the ROI mask $m_t$​, the overall coding framework is achieved by optimizing the following rate-distortion tradeoff:

Video compression uses rate-distortion optimization to balance compression ratio and video quality, aiming to minimize the bitrate of the encoded video and the distortion. Similar to [xxx], we use a factor $\beta$ to control the reconstruction quality of ROI and non-ROI regions. Given the ROI mask $m_t$, the overall coding framework is optimized by the following rate-distortion tradeoff:
$$
L_t=\lambda D_t+R_t=\lambda [d_{ROI} + d_{non-ROI}]+[H(\hat{y}_t)+H(\hat{g}_t)]\\
d_{ROI} = d\big(m_t\odot(x_t,\hat{x}_t)\big),~
d_{non-ROI} = \beta d\big((1-m_t)\odot(x_t,\hat{x}_t)\big)
$$
$L_t$ is the total loss function for current time step $t$, which comprises two parts: distortion term $D_t$ and bit rates term $R_t$. $d(x_t, \hat{x}_t)$ measures the distortion between original frame $x_t$ and reconstructed frame $\hat{x}_t$. Distortion can be quantified by MSE(mean squared error) and  MS-SSIM (multiscale structural similarity). $\lambda$ is a Lagrange multiplier that controls the trade-off between distortion $D_t$ versus rate $R_t$. $H(\hat{y_t})$ and $H(\hat{g}_t)$ represents the bit rates of latent representations $\hat{y}_t$ and $\hat{g}_t$.

为了弥补替换GDN后造成的Gaussianization, denoising, and sampling能力的缺失，我们对学生网络与教师网络之间的mv编码器和contextual编码器输出的bitstream的分布计算Kullback–Leibler (KL) divergence，并作为蒸馏损失 $L_{KD}$​ ，来转移全精度网络模型的知识。

To compensate for the lack of gaussianization, denoising, and sampling ability caused by replacing GDN, we compute the Kullback-Leibler (KL) divergence over the distribution of the bitstreams output by the mv encoder and contextual encoder between the student and teacher networks, and use it as a distillation loss $L_{KD}$, to transfer the knowledge from full-precision teacher network to low-precision student.

| Stage   | Training Modules |                      Loss                      | Steps | LR        | Frames |
| ------- | ---------------- | :--------------------------------------------: | ----- | --------- | ------ |
| Stage 1 | MV-E/D           |      $\lambda D_{warp}+ R_{mv} + L_{KD}$       | 250k  | $1E^{-4}$ | 2      |
| Stage 2 | MV-E/D, ME       |       $\lambda D_{warp}+R_{mv}+ L_{KD}$        | 200k  | $1E^{-4}$ | 2      |
| Stage 3 | Contextual-E/D   |         $\lambda D_{recon}+R+ L_{KD}$          | 300k  | $1E^{-4}$ | 2      |
| Stage 4 | All              |         $\lambda D_{recon}+R+ L_{KD}$          | 400k  | $1E^{-4}$ | 2      |
| Stage 5 | All              | $\frac{1}{T}\sum^T{(\lambda D_t+R^t+ L_{KD})}$ | 800k  | $1E^{-4}$ | 5      |

我们使用渐进式训练策略训练整个编码框架，分为5个阶段，如表x所示。首先，我们固定其他网络模块参数，只训练motion vector(MV) encoder(E)/decoder(D)模块；之后，联合训练MV-E/D和motion estimation(ME)；之后，固定MV-E/D和MV网络参数，训练Contextual E/D模块；最后，开放网络所有参数，整个网络联合以端到端方式训练。

We train the entire coding framework using a progressive training strategy, which has 5 stages as shown in Table x. First, we fix other module parameters and train motion vector(MV) encoder(E)/decoder(D) module. Then, MV-E/D and motion estimation(ME) were jointly trained. Then, the Contextual E/D module was trained by fixing the MV-E/D and MV network parameters. Finally, all parameters of the network were opened, and the whole network was jointly trained in an end-to-end manner.

在前两阶段，loss函数由参考帧 $\hat{x}_{t-1}$​ 经过重构mv的warp后引起的失真与mv的bit rates构成。

In the first two stages, the loss function consists of the distortion caused by the warping of the reference frame $\hat{x}_{t-1}$ by reconstructed mv and the bit rates for compressing mv.
$$
L_t = \lambda D_{warp} + R_{mv} = \lambda [d_{ROI}(x_t, \ddot{x}_t)+d_{non-ROI}(x_t, \ddot{x}_t)] + H(\hat{g}_t) + L_{KD}
$$

> loss including the distortion of the reference frame warped by the compressed motion and the bit-rate for compressing mv。

第3阶段之后，MV的估计和编解码网络已收敛，我们固定这一部分参数，对条件编解码的部分参数进行训练。loss函数由重构帧和原始帧之间的失真以及所有latent code的bit rates组成。

After stage 2, MV-E/D and ME have converged, we fix these modules' parameters and train the contextual codec part. The loss function consists of the distortion between the reconstructed frame and the original frame and the bit rates of all latent codes.
$$
L_t = \lambda D_{recon} + R = \lambda [d_{ROI}(x_t, \hat{x}_t)+d_{non-ROI}(x_t, \hat{x}_t)] + [H(\hat{g}_t)+ H(\hat{y}_t)] + L_{KD}
$$

前4阶段只采用两个连续时间步的帧进行训练，忽略了 $\hat{x}_t$ 对于下一帧 $x_{t+1}$ 的潜在误差影响，从而导致导致误差传播。因此在最后一个阶段，将 $\hat{x}_t$ 作为 $x_{t+1}$​ 编码阶段的参考帧，获得多个连续时间间隔的损失，其中T表示时间间隔，实验中我们设置为5.

The first four stages only use frames of two consecutive time steps for training, ignoring potential error impact of $\hat{x}_t$ to the next frame $x_{t+1}$, which may leads to error propagation. So in the last stage, $\hat{x}_t$ is used as the reference frame in $x_{t+1}$ encoding process to obtain the loss of multiple consecutive time intervals, where T denotes the time interval, we set as 5 in our experiments.
$$
L^T=\frac{1}{T}\sum_{t=1}^TL_t=\frac{1}{T}\sum_{t=1}^T\{\lambda [d_{ROI}+d_{non-ROI}]+[H(\hat{g}_t)+ H(\hat{y}_t)]+L_{KD}\}
$$




[^1]: H. Sun, L. Yu, and J. Katto, “Learned image compression with fixed-point arithmetic,” in Proc. Picture Coding Symposium, 2021.
[^2]: End-to-end learned image compression with fixed point weight quantization.



[^3]:W. Hong, T. Chen, M. Lu, S. Pu, and Z. Ma, “Efficient neural image decoding via fixed-point inference,” IEEE Trans. Circuits and Systems for Video Technology, vol. 31, no. 9, pp. 3618–3630, 2020.
[^4]: H. Sun, L. Yu, and J. Katto, “Q-lic: Quantizing learned image compression with channel splitting,” IEEE Trans. Circuits and Systems for Video Technology, 2022 (early access).
[^5]: End-to-end learned image compression with quantized weights and activations



## Experiments



### Experiments Setup

**Datasets** training dataset是Vimeo-90K septuplet dataset[].Vimeo-90K is a large-scale high-quality video dataset for lower-level video processing, contrains 89,800 video clips covering a variety of scenes and actions,  each clip consisting of seven frames. 将视频帧随机裁剪得到 256 $\times$ 256大小的patch输入网络。

测试数据集是 来自common test conditions[]的HEVC Standard Test Sequences Class B (1080P), C (480P), D (240P), E (720P) ，以及MCL-JCV dataset(1080p)和UVG dataset(1080p)。

Test dataset is HEVC Standard Test Sequences Class B (1080P), C (480P), D (240P), E (720P) from the common test conditions. As well as MCL-JCV dataset(1080p) and UVG dataset(1080p).

 

**Implementation Details.** 我们的深度编码模型基于DCVC条件编码模型框架，并基于UNISAL模型生成显著性mask图，得到帧中的ROI区域和non-ROI区域。正如之前提到的，我们采用渐进式的训练框架，使用Adam优化器xxx，$\beta_1$ 和 $\beta_2$ 分别为 0.9 and β2 as 0.999，分为5个训练阶段，每个阶段的初始learning rate均设置为1e-4，损失趋于稳定后learning rate设置为1e-5. 训练batch size设置为4. 我们训练4个模型with different $\lambda$ (MSE:256,512,1024,2048; MS-SSIM: 8, 16, 32, 64)，对应不同的QP值(QP = 37, 32, 27, 22). 对于控制ROI和non-ROI区域的码率和重建质量的因子 $\beta$ ，实验中我们设置为0.5. 我们基于LSQ方法进行实验。我们采用PyTorch，在2张 NVIDIA RTX 3090 GPU上进行所有实验。

Our learned compression model is based on DCVC framework and generates saliency map using UNISAL model to obtain the ROI region and non-ROI region. As mentioned before, we adopt a progressive training scheme with 5 training phases using Adam optimizer xxx, $\beta_1$ and $\beta_2$ is set as 0.9  and 0.999, respectively; and initial learning rate of each phase is set to 1e-4. After the loss becomes stable, the learning rate is set to 1e-5. The training batch size is set to 4. We train four models with different $\lambda$(MSE: 256,512,1,024,2048; MS-SSIM: 8, 16, 32, 64), corresponding to different QP values (QP = 37, 32, 27, 22). For the factor $\beta$, which controls the bit rate and reconstruction quality in ROI and non-ROI regions, we set it to 0.5 in the experiments. Our models are implemented in PyTorch and trained on two NVIDIA RTX 3090 GPUs.

**Testing settings** The GOP (group of pictures) size is same with [4], namely 10 for HEVC videos and 12 for non-HEVC videos. As this paper only focuses on inter frame coding, for intra frame coding, we directly use existing deep image compression models provided by CompressAI [36]. We use cheng2020-anchor [20] for MSE target and use hyperprior [13] for MS-SSIM target.

Evaluation Metrics.  Both PSNR and MS-SSIM [30] are used to measure the quality of the reconstructed frames in comparison to the original frames.  Bits per pixel (bpp) is used to measure the number of bits for encoding the repre- sentations including MVD and residual.



**Bit-width candidates**

我们为ROI区域和non-ROI区域分别设置了两组quantization bit-width candidates，lower set {non-ROI: 2,3,4; ROI: 4,5,6} with target bit-width 4 and higher set {non-ROI: 6,7,8; ROI: 8.9,10} with target bit-width 8。对于权重，我们采用固定的bit-width，和activations的target bit-width保持一致。

We set two sets of quantization bit-width candidates for ROI region and non-ROI region respectively, lower set {non-ROI: 2,3,4; ROI: 4,5,6} with target bit-width 4 and higher set {non-ROI: 5,6,7; ROI: 7.8,9} with target bit-width 8. For the weights, we use fixed bit-width, which is the same as the target bit-width of activations.

### Results and Analysis

**R-D Performance**

在图xx和图xx中，展示了我们方法的率失真曲线。图xx的失真基于PSNR，图xx的失真基于MS-SSIM指标。观察图可发现，当采用静态的LSQ量化方法时，8-bit和4-bit模型都出现了不同程度的性能降低，在4-bit量化模型上更为明显。在对模型分别进行4-bit和8-bit的动态bit-width分配后，我们可以发现，在HEVC-E数据集上，当bpp为0.034的情况下，4-bit dynamic model相比于4-bit static model 取得了 1.4db的增益。相同重建质量情况下，对于1080p数据集(MCL-JCV, UVG, HEVC-B)，动态的4bit/8bit方法相比于静态的量化，码率节省分别为 18.2%, 17.9%, 19.4%/12.3%, 11.8%, 13.6%; 其他分辨率数据集上(HEVC-C, HEVC-D, HEVC-E), 4bit/8bit的动态量化相比于静态也有提升，码率节省分别为 9.5%, 11.2%, 12.4%/8.2%, 10.4%, 11.8%. 通过R-D图中的曲线我们可以发现，在不同分辨率和不同类型的视频数据集上，我们的动态量化方法都有明显的性能提升。 

Figures xx and xx present the rate-distortion curves of our method. The distortion in Figure xx is based on PSNR, while in Figure xx is based on MS-SSIM metric. Observations from the figures indicate that when using the static LSQ quantization method, both the 8-bit and 4-bit models exhibit varying degrees of performance degradation, with more pronounced effects in the 4-bit model. After applying dynamic bit-width allocation to the models at 4-bit and 8-bit, we observe that on the HEVC-E dataset, at bpp of 0.034, the 4-bit dynamic model reachs a 1.4 dB gain compared to the 4-bit static model. With the same reconstruction quality, for the 1080p datasets (MCL-JCV, UVG, HEVC-B), the dynamic 4-bit/8-bit methods achieving 18.2%, 17.9%, 19.4%/12.3%, 11.8%, 13.6% bitrates saving respectively compared to static quantization. On other resolution datasets (HEVC-C, HEVC-D, HEVC-E), dynamic quantization in 4-bit/8-bit also shows improvements, with bitrate saving of 9.5%, 11.2%, 12.4%/8.2%, 10.4%, 11.8% respectively. The R-D curves demonstrate significant performance enhancements of our proposed method on different resolutions and types of video clips.



我们对量化后ROI区域和non-ROI区域的重建性能进行了比较。如图xxx所示。图中展示了32-bit baseline, 8-bit的动态量化，quantized ROI区域和non-ROI区域的R-D曲线。我们可以观察到，虽然整体图像上的重建质量在动态量化后出现了轻微降低，但ROI区域的性能和全精度下的接近，而在人眼容易忽视的非显著区域上，性能略微下降。说明bit-allocator对不同的区域的计算资源进行了精确分配，对显著前景区域分配了更多计算bit。总体而言，proposed method可以在ROI区域实现great的coding gain，有利于提升人眼对于实时视频的感知质量。



We compare the reconstruction performance of quantized ROI regions and non-ROI regions. As shown in Figure xxx. The figure illustrates the R-D curves for 32-bit baseline, 8-bit dynamic quantization, quantized ROI regions, and non-ROI regions. We observed that although reconstruction quality on the whole image is slightly degraded after dynamic quantization, the performance of ROI region is close to full precision model; and the performance is  in non-salient regions that are easily overlooked by the human eye. This indicates that the bit-allocator precisely allocated computational resources to different areas, assigning more computational bits to significant foreground areas and less to unimportant background regions. In general, our proposed method achieves substantial coding gains in the ROI areas, enhancing the perceptual quality of real-time video for human eyes.



 

**Complexity Analysis**

在standard test video datasets上的quantization results 如表xxx所示。将FLOPs作为计算复杂度的度量。由于不同区域具有不同的eye fixation saliency，ROI区域和non-ROI区域是采用不同的计算资源进行处理的，所以能更有效地用于低延迟场景视频编码的高效推理。从表xxx中我们可以清晰地看出，与32-bit baseline相比，4-bit动态量化的模型实现了89.2%的FLOPs减少，且仅出现了较小程度的bitrate increase。相比于4-bit/8-bit的静态量化model，我们的动态模型减少了大概20%的FLOPs，且很好地减少了重建帧的失真。

The quantization results on standard test video datasets are shown in Table xxx. We use FLOPs as a metric of computational complexity. Since different regions have different eye fixation saliency, ROI and non-ROI regions are processed with different computing resources, allowing for more efficient inference for video coding in low-latency scenarios. From Table xxx, we can clearly see that the model with 4-bit dynamic quantization achieves 89.2% reduction in FLOPs with only a slight bitrate increase compared to the 32-bit baseline. Compared with the 4-bit/8-bit static quantization model, our dynamic model reduces FLOPs by about 20% and reduces the distortion of reconstructed frames well.



### Qualitative Results

为了观察bit-allocator所制定的量化策略，我们对每个video clip的输入帧分配的bit-width进行可视化。如图xxx所示。Bit-allocator针对不同的视频内容进行了准确分配，具体来说，ROI区域为复杂图像，人像或高运动的场景的情况下，将被分配更高的bit-width来进行高质量的重建；而光流变化较小的区域，以及整体图像结构较为简单的帧，则会被分配较低的bit-width来节省计算资源。高复杂度区域保留较多计算资源，有利于减少量化误差的累积导致的重建帧失真。

To observe the quantization strategy developed by bit-allocator, we visualize the bit-width assigned to the input frame of each video clip. As shown in Figure xxx. The Bit-allocator accurately allocates different video contents. Specifically, in the case of complex images, portraits or high motion scenes, a higher bit-width will be allocated for high-quality reconstruction. The regions with small optical flow variation or simple structure, will be assigned a lower bit-width to save computing resources. High complexity regions retain more computing resources, which is beneficial to reduce the aggravation of reconstructed frame distortion caused by the accumulation of quantization errors.

 

## Conclusion

In this work, we introduces a ROI-aware dynamic quantization technique for neural video compression networks that dynamically adjusts quantization strategies based on the ROIs in video frames, assigning higher bit widths to key areas and lower bit widths to less critical areas. We developed an efficient bit allocator that adaptively sets the quantization levels for different regions in frames, guided by the motion and image content complexity. Experimental results on standard test videos demonstrate the effectiveness of proposed dynamic quantization strategy.





本文提出了一种ROI(感兴趣区域)感知的NVC网络动态量化方法，分析视频帧中的ROI并动态改变量化策略，为关键区域分配较高的比特宽度，为不太关键的区域分配较低的比特宽度。本文提出一种有效的比特分配器，在区域的运动和纹理复杂度的指导下，自适应地确定不同区域和帧的量化水平。在标准测试视频上的实验结果验证了所提出的动态量化方法的有效性。与静态的低比特量化方法相比，所提方法在减少20%FLOPs的情况下，实现了bitrate的有效节省，在部分数据集上可达到19%；除此之外，ROI区域的高质量重建也提升了人眼的视觉感知效果。

 

本文提出一种可变比特率、基于感兴趣区域的神经视频编解码器。据我们所知，这是第一个可以动态调整全局(每帧)和局部(帧内)比特分配的神经视频编解码器。这是通过引入两个控制参数来实现的，一个控制现有多码率模型中的码率惩罚，另一个控制ROI区域和非ROI区域失真惩罚之间的权衡。由此产生的编解码器能够在精细分辨率上进行速率和质量控制，这有可能实现实际用例，例如以最小ROI质量的固定速率编码。



在感兴趣的地区证明了BD-rate的大幅节省，在某些情况下实现了60%以上的节省。在电话会议环境中，基于roi的朴素编码的好处是有限的，并对如何缓解这个问题提供了直观的看法。

 

(a)第一个基于roi的多速率神经视频编解码器，可以动态控制和调整局部(帧内)和全局(每帧)比特率分配。

(b)对常见视频数据集的定性和定量评估，表明感兴趣区域的R-D性能有很大提高，BD-rate节省高达60%。

(c)有证据表明，像电话会议这样的用例不能从简单的ROI编码中受益，而且在具有高运动内容的视频中，通过使用更智能的码率保真控制算法，可以获得更大的增益。





高运动内容！！！







| Model         | Size(MB) |      | FLOPs(G) | bitrate increase |
| ------------- | -------- | ---- | -------- | ---------------- |
| 32-bit        | 52.4     |      | 2225     | 0.0%             |
| 8-bit static  | 14.4     |      | 542      | 18.6%            |
| 4-bit static  | 8.3      |      | 282      | 27.3%            |
| 8-bit dynamic | 13.6     |      | 462      | 7.2%             |
| 4-bit dynamic | 7.8      |      | 241      | 14.1%            |





在标准测试视频(HEVC, UVG, mc - jcv)上的实验验证了所提方法的有效性。实验结果表明，我们所提出的ROI-aware dynamic quantization在learned video compression networks上表现出了出色的性能。





To select the appropriate bit-width, we employ a lightweight bit-allocator that determines bit-width based on the complexity of region features. Specifically, we configure separate bit-allocators for ROI and non-ROI regions.. ROI regions are allocated higher bit-width compared to non-ROI regions. Inspired by \cite{hong2022cadyq}, we formulate quantization strategies for the two regions in terms of structure complexity and motion complexity within the frame, respectively. Bit-allocators output a probability vector consisting of probability $\pi^k(\boldsymbol{x})$ for each bit-width candidate:





**补充一句话：已有的基于语义的编码大多是针对预训练的语义分割、目标检测等网络，在视频内容复杂多样的情况下，无法标注庞大的对象的语义信息，因此我们采用人眼的视觉显著性来进行区分，**
