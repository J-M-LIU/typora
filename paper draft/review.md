## Reviewer #1

### Questions

- 1. Paper Summary:

  - The paper proposes a method for neural compression that is saliency aware. The method computes salient regions using the UNISAL network which outputs dense saliency maps. The saliency maps are used to allocate bits to highly salient regions using different quantization levels, in other words, salient regions are less quantized than non-salient regions. The bit widths are chosen from a set for the salient and non-salient regions and the choice is made by an allocator which uses local complexity (measured by features and flow) to ensure that high complexity regions are allocated high enough bit widths. The neural compression network is trained to jointly optimize bitrate and distortion and is tested on a reasonable set of benchmarks.

- 2. Paper Strengths:

  - \* The idea to allocate different bit-widths to salient and non-salient regions is interesting and well motivated 
    \* The results seem good

- 3. Paper Weaknesses:

  - Missing comparisons to other neural compression methods
    Test with other saliency methods
  - 缺失与其他神经压缩方法的比较
    使用其他显著性方法进行测试

- 4. Overall Recommendation:

  - Weak Accept

- 5. Justification For Recommendation And Suggestions For Rebuttal:

  - This is a interesting idea and a new spin on the saliency with neural compression angle. The use of saliency by itself is interesting and useful especially for bandwidth constrained situations. The bit allocator is a good use of saliency, where network activations are less quantized in the salient regions and mimics the way saliency is used in traditional codecs. In addition to being interesting the method does appear to work well with the 4bit dynamic scoring similarly to the 8bit static in Fig 2. For these reason I am voting accept.

    There are a couple of things missing. The first is that only UNISAL was tested as the saliency model. There are many saliency models and they are all competitive, an ablation on this would have been interesting. The next is that there are no comparisons provided with other neural compression models which makes it difficult to measure the contribution of this method. This is particularly difficult for neural compression where we like to see RD curves and everyone scales their charts slightly differently, so if I was going to look at other papers to make the comparison I would need to be extremely careful in interpreting the charts.

    这是一个有趣的想法，也是神经压缩角度对显著性的一个新的诠释。显著性的使用本身是有趣和有用的，特别是在带宽受限的情况下。比特分配器很好地利用了显著性，其中显著区域的网络激活较少量化，并模仿了传统编解码器中使用显著性的方式。除了有趣之外，该方法似乎与图2中的8位静态评分类似的4位动态评分很好地工作。出于这些原因，我投票接受。

    少了几样东西。第一，只有UNISAL被测试为显著性模型。有许多显著性模型，它们都是有竞争力的，对此进行消融会很有趣。其次，没有与其他神经压缩模型进行比较，这使得很难衡量该方法的贡献。这对于神经压缩来说尤其困难，因为我们喜欢看到RD曲线，每个人的图表缩放略有不同，所以如果我要看其他论文进行比较，我需要非常小心地解释图表。

## Reviewer #4

### Questions

- 1. Paper Summary:

  - A method is proposed for adaptive quantization of weights and activations for neural video compression. This is accomplished by dynamic bit allocation based on saliency. Experimentation is performed to demonstrate the rate-distortion improvement over static quantization methods.

- 2. Paper Strengths:

  - Neural video compression is an active area of research with interesting applications. Adaptive quantization to reduce complexity, memory footprint, and bandwidth is a relevant contribution. 

- 3. Paper Weaknesses:

  - 1. Several technical details are not fully explained. These are as follows:

       a. Sec 3.1: "\alpha is a predefined or learnable upper bound". In the actual implementation, is it predefined or learnable? If predefined, how were the values chosen? If learnable, what were the learnt values?

       b. Is the bit-width selection for ROI/non-ROI regions is done on a per-frame basis, or on smaller regions? What if a frame has multiple ROI regions? Is the same bit-width selected for all ROI regions within a frame? How are the bit-widths used to encode a frame and MVs communicated to the decoder?

       c. Specifics on the calculation of \sigma(x), \sigma(v), G(x), G(v) in Eq. (5) should be provided. Are these calculated for each pixel within each region, or a single value calculated for each region? What are the window sizes used to calculate these values? These parameter choices could have a significant impact on the actual performance 

    2. Bit allocation based on saliency is a very well-studied topic. There are several relevant papers in this area that have not been considered and referred to. For instance,

       a. Ku, ChungWen, Guoqing Xiang, Feng Qi, Wei Yan, Yuan Li, and Xiaodong Xie. "Bit allocation based on visual saliency in HEVC." In 2019 IEEE Visual Communications and Image Processing (VCIP), pp. 1-4. IEEE, 2019.

       b. Hadizadeh, Hadi, and Ivan V. Bajić. "Saliency-aware video compression." IEEE Transactions on Image Processing 23, no. 1 (2013): 19-33.

       c. Zhu, Shiping, Qinyao Chang, and Qinghai Li. "Video saliency aware intelligent HD video compression with the improvement of visual quality and the reduction of coding complexity." Neural Computing and Applications 34, no. 10 (2022): 7955-7974.

    3. Experimental section is weak. 

       a. No comparison against standard codecs such as HEVC and/or VVC is provided.

       b. For the results in Section 4.2, the authors should provide values for the BD-BR metric which is the standard way of evaluating and comparing codec performance.

       c. The evaluation approach for complexity is not very clear. How are the FLOP values calculated for different bit-widths? For instance, is it assumed that eight 4-bit operations is equivalent to 1 FLOP? How would this dynamic quantization be implemented on an actual GPU? Are the values actually converted to fp8, fp4, or int8, int4, etc.? What about values like 5-bit or 6-bit as indicated in the results in Fig. 5? There is no native support for such bit-widths in a GPU, and so an actual implementation will likely require representing in fp32 (even though the value itself might be quantized to a lower number of bits) which wouldn't result in any speed up in operation. Hence, I would suggest that the authors report not just flop counts, but also improvement in actual run-times to support their claims of efficiency. Also, please provide details on "ROI and non-ROI regions are allocated different computing resources." 

    4. Some minor typos:

       a. p.5 "Similarly, activation quantization process can be …" should be "Similarly, weight quantization process can be …" 

       b. p.6, Sec 3.2 "devide ROI/non-ROI" --> "divide ROI/non-ROI"

- 4. Overall Recommendation:

  - Weak Reject

- 5. Justification For Recommendation And Suggestions For Rebuttal:

  - For the rebuttal, it would be good if the authors could provide the technical details listed above, and also add comparisons against HEVC.

  1. 有几个技术细节没有完全解释。如下所示。

     a.第3.1节:“\alpha是预定义的或可学习的上限”。在实际实现中，它是预定义的还是可学习的?如果是预定义的，如何选择值?如果可以学习，那么学习到的价值观是什么? **直接解释是引入LSQ的可学习的alpha值**

     b. ROI/非ROI区域的比特宽度选择是基于每帧进行的，还是针对较小的区域进行的?如果一帧有多个ROI区域呢?一帧中所有ROI区域是否选择相同的位宽? bit-widths如何传递给解码器？ **因为人眼的关注特性，在连续的帧中关注区域只有一个。关于如何传递额外的位宽信息：这个研究一下**

     c.提供式(5)中\sigma (x)、\sigma (v)、G(x)、G(v)的具体计算方法。这些值是针对每个区域中的每个像素计算的，还是针对每个区域计算的单个值?计算这些值时使用的窗口大小是多少?这些参数的选择可能会对实际性能产生重大影响 

  2. 基于显著性的比特分配是一个非常深入的研究课题。在这方面还有几篇相关论文没有被考虑和参考。例如，

      顾忠文，项国庆，祁峰，闫伟，李媛，谢晓东。" HEVC中基于视觉显著性的比特分配"2019年IEEE视觉通信和图像处理(VCIP)，第1-4页。Ieee, 2019。
      b.哈迪扎德、哈迪和伊万V. Bajić。“显著性感知的视频压缩。”IEEE图像处理汇刊第23期。1(2013): 19-33。
      c.朱世平，常勤尧，李青海。视频显著性感知的智能高清视频压缩随着视觉质量的提高和编码复杂度的降低。神经网络计算与应用10(2022): 7955-7974。
  3. 实验部分薄弱。 

     a. 没有提供与标准编解码器(如HEVC和/或VVC)的比较。

     b. 对于4.2节中的结果，作者应该提供BD-BR度量的值，这是评估和比较编解码器性能的标准方法。

     

     ![image-20240624233954816](/Users/liujiamin/Library/Application Support/typora-user-images/image-20240624233954816.png)

     c. 复杂性的评价方法不是很清楚。如何为不同的位宽计算flops?例如，是否假设8次4位操作相当于1次发牌?如何在实际的GPU上实现这种动态量化?这些值是否实际转换为fp8、fp4或int8、int4等?在图5的结果中，像5位或6位这样的值呢?GPU中没有对这种位宽的原生支持，因此实际实现可能需要用fp32表示(即使值本身可能被量化为更低的位数)，这不会导致任何操作速度的提高。因此，我建议作者不仅要报告失败次数，还要报告实际运行时的改进，以支持他们所声称的效率。另外，请详细说明“ROI和非ROI区域分配不同的计算资源”。 **Bit-FLOPs 模拟 The Bit-FLOPs is calculated as the product of the weight bit-width, the activation bit-width and the FLOPs (the multiplication and add operations).**

     **补充1. Bit-FLOPs的计算方式**

     **补充2. 补充实际GPU推理的时间：{2,4}{4,8} {4,8}{8,16}**

     **补充3. BD-Bitrate的表**

     **补充4. 提供与HEVC、VVC等的对比**

     **补充5. 关于图像复杂度和光流复杂度的计算**

     

  

  

  

  3. 一些小的打字错误:

  a. p.5“同样，激活量化过程可以是……”应该是“同样，权重量化过程可以是……”

  b. p.6，第3.2节“划分ROI/非ROI”—＆gt;“划分ROI/非ROI”

  - 4。总体建议:

    -弱拒绝

  - 5。推荐理由和反驳建议:

    -对于反驳，如果作者可以提供上面列出的技术细节，并与HEVC进行比较，那就太好了。

## Reviewer #5

### Questions

- 1. Paper Summary:

  - The paper presents a novel approach to neural video compression (NVC) by introducing a Region of Interest (ROI)-aware dynamic quantization method to address the high computational complexity of deep neural networks in low-latency scenarios. Traditional NVC methods often employ a fixed bit-width quantization strategy, which does not consider the unique characteristics of video frames. The proposed method, however, dynamically alters the quantization strategy based on the analysis of ROIs within video frames, allocating higher bit-widths to critical regions and lower bit-widths to less crucial areas.

- 2. Paper Strengths:

  - (1) One of the significant strengths of the proposed method is its ability to dynamically adjust the quantization strategy according to the content of the video frames.

    (2) The paper introduces an efficient bit-allocator that adaptively determines the quantization levels for different regions within video frames. 

    (3) The authors provide a thorough experimental evaluation of their method using standard test videos. The results demonstrate clear improvements in rate-distortion performance over static quantization methods, with significant gains in both PSNR and MS-SSIM metrics. 

- 3. Paper Weaknesses:

  - （1） This article lacks comparison with previous methods.
    （2）The article does not provide analysis on the execution time of the program or its computational complexity.

  (1)缺乏与以往方法的对比。
  (2)文中没有对程序的执行时间及其计算复杂度进行分析。

- 4. Overall Recommendation:

  - Accept

- 5. Justification For Recommendation And Suggestions For Rebuttal:

  - The paper has good writing and clear representation.







梯度：图像：转化为单通道，水平和垂直方向的梯度均值，B 2

光流：B  2 H W 两个channel 分别表示水平和垂直方向 ——> B 2

标准差：图像：通道方向上的       B C

光流：通道方向上的 B 2

**梯度**

对于输入图像，转化为灰度图像，计算水平和垂直方向的梯度幅值，根据saliency mask分别可得ROI和Non-ROI区域的梯度均值。B x 2

光流：获取光流在水平和垂直channel上的梯度幅值，B x 2

**标准差**

图像：根据saliency mask得到两个区域的通道级别的标准差 B x C

光流：通道级别的标准差 B x 2



首先计算图像和光流在通道级别的标准差。之后，输入图像转化为灰度图，计算水平和垂直方向的平均梯度幅值；对于光流则分别计算两个通道上的区域平均梯度。整体过程均基于saliency mask来获取区域内的计算结果。

$\sigma(x)$, $\sigma(v)$ represent the channel level standard deviation of input feature and optical flow.within a  The input feature is converted to gray-scale image to calculate average magnitude of gradient in horizontal and vertical directions. And $G(v)$ measures the average gradients on the two channels of optical flow.

And for the optical flow, the average gradients are calculated for the two channels respectively. 

对于光流计算其两个通道上的平均梯度。







First, calculate the standard deviation of the image and optical flow at the channel level. After that, the input image is converted to a grayscale image, and the average gradient magnitudes in the horizontal and vertical directions are calculated; for the optical flow, the regional average gradients are calculated for the two channels respectively. The overall process obtains the calculation results within the region according to the saliency mask.





Given an region feature $x_t^R$ with framde index $t$, 





为了选择合适的位宽度，采用了一个轻量级的位分配器，该分配器根据区域特征的复杂度来分配位宽度。具体来说，我们分别为ROI区域和非ROI区域配置了两个比特分配器。ROI区域比非ROI区域具有更高的候选位宽。受\cite{hong2022cadyq}的启发，我们根据帧内结构复杂度和运动复杂度分别制定了两个区域的量化策略。位分配器为每个位宽候选输出一个由概率$\pi^k(\boldsymbol{x})$组成的概率向量:


$$
\pi(\boldsymbol{x_t^R}) = \mathrm{Softmax}{\Big(fc\big(\sigma(\boldsymbol{x_t^R}),G(\boldsymbol{x_t^R}),\sigma(\boldsymbol{v_t^R}),G(\boldsymbol{v_t^R})\big)\Big)},
$$


其中$\sigma(\boldsymbol{x})$和$\sigma(\boldsymbol{v})$分别表示图像的区域特征和预测光流的标准差。$G(\boldsymbol{x})$和$G(\boldsymbol{v})$分别表示区域特征的梯度和光流向量在水平和垂直方向上的梯度。这几个指标评估了图像的纹理结构复杂度和运动复杂度，具有复杂结构或高运动内容的区域将被分配更高的位宽。我们将这些指示符连接到全连接层$fc$，并获得输出logits $\pi^1,\pi^2,... ,\pi^k$，对应于K位宽候选。分配概率最大的量化位宽由:

 $\sigma(\boldsymbol{x_t^R})$ and $\sigma(\boldsymbol{v_t^R})$ denote the channel-level standard deviations for the input feature and optical flow.  $G(\boldsymbol{x_t^R})$ measures the average magnitudes of gradients in both horizontal and vertical directions of grayscale image of $\boldsymbol{x_t^R}$. And $G(\boldsymbol{v_t^R})$ represents the average gradients across the two channels of optical flow. Subsequently, these metrics are concatenated and input into a fully connected layer $f c: \mathbb{R}^{3+2+2+2}\Rightarrow\mathbb{R}^K$, where K denotes the number of bit-width candidates. The output logits of bit-selector are $\pi^1,\pi^2,... ,\pi^k$. The quantization bit-width with the largest allocation probability is selected by:



The input feature is transformed to a grayscale image to compute average magnitudes of gradients in both horizontal and vertical directions.




$$
\boldsymbol{x}_{t,q}^R =Q_{b^k}(\boldsymbol{x}_{t,q}^R)=\arg\max_{Q_{b^k}(\boldsymbol{x})}\pi^k(\boldsymbol{x}_{t,q}^R).
$$

$$
P^k(\boldsymbol{x}_t^R)=\frac{\exp\left((\log\pi^k(\boldsymbol{x}_t^R)+g^k)/\tau\right)}{\sum_{j\in\Omega}\exp\left((\log\pi^j(\boldsymbol{x}_t^R)+g^j)/\tau\right)},
$$


The bit-width selection is done on per-frame basis, and we use two 16-bit integers for bypass encoding, writing them directly into the binary bitstream.



删除3.2 Moreover, due to the continuity between frames in a video clip, when minimal changes occur between frames, the quantization strategy of the current frame can leverage that of previously transmitted frames to enhance coding efficiency.





Our learned compression model is applied directly to  contextual video compression framework, including DCVC-DC, DCVC-HEM and DCVC. We utilizes the UNISAL model to generate saliency maps, identifying ROI and non-ROI regions. As previously described, we employ a progressive training scheme with 5 training phases. We use Adam optimizer with $\beta_1$ and $\beta_2$ is set as 0.9 and 0.999 respectively. The initial learning rate of each phase is set to 1e-4. After the loss becomes stable, the learning rate is set to 1e-5. The training batch size is set to 4. We train four models with different $\lambda$ (MSE: 256, 512, 1024, 2048; MS-SSIM: 8, 16, 32, 64), corresponding to different QP values (QP = 37, 32, 27, 22). For the factor $\beta$, which controls the bitrates and reconstruction quality of ROI and non-ROI regions, we set it to 0.5 in our experiments. Our models are implemented in PyTorch \cite{paszke2019pytorch} and trained on two NVIDIA RTX 3090 GPUs.





Fig. \ref{fig2} and Fig. \ref{fig3} present the rate-distortion curves of our method, with distortion Fig. \ref{fig2} measured by PSNR and in Fig. \ref{fig3} by the MS-SSIM metric. Observations from the figures indicate that when using the static LSQ quantization method leads to performance degradation in both the 8-bit and 4-bit models, more noticeably in the 4-bit model. Upon implementing dynamic bit-width allocation in the 4-bit and 8-bit models, significant improvements are observed. For instance, on the HEVC-E dataset at a bitrate of 0.034 bits per pixel (bpp), the 4-bit dynamic model achieves a 1.4 dB gain compared to the static model. Additionally, With the same reconstruction quality, in the 1080p datasets (MCL-JCV, UVG, HEVC-B), the dynamic 4-bit/8-bit methods achieve bitrate savings of 18.2\%, 17.9\%, 19.4\%/12.3\%, 11.8\%, 13.6\% respectively, when compared to static quantization. On other resolution datasets (HEVC-C, HEVC-D, HEVC-E), dynamic quantization in 4-bit/8-bit also shows improvements, with bitrates saving of 9.5\%, 11.2\%, 12.4\%/8.2\%, 10.4\%, 11.8\% respectively. The R-D curves demonstrate significant performance enhancements of our proposed method on different resolutions and types of video content.

图\ref{fig2}和图\ref{fig3}展示了我们方法的率失真曲线，失真图\ref{fig2}由PSNR测量，图\ref{fig3}由MS-SSIM度量。从图中可以看出，当使用静态LSQ量化方法时，在8位和4位模型中都会导致性能下降，在4位模型中更明显。在4位和8位模型中实现动态位宽分配后，性能得到显著改善。例如，在HEVC-E数据集上，以0.034比特/像素(bpp)的比特率，4比特动态模型与静态模型相比实现了1.4 dB的增益。在1080p数据集(mc - jcv, UVG, HEVC-B)上，在重构质量相同的情况下，动态8位/4位量化方法比静态量化方法码率分别节省18.2％，17.9％，19.4％/12.3％，11.8％，13.6％。在其他分辨率数据集(HEVC-C, HEVC-D, HEVC-E)上，4位/8位的动态量化效果也有所提升，码率分别节省9.5％，11.2％，12.4％/8.2％，10.4％，11.8％。R-D曲线证明了所提出方法在不同分辨率和类型的视频内容上的显著性能增强。

我们发现，动态量化方式只出现了小幅度的bitrate increase。

table xx shows the BR-rate comprasion with PSNR metrics. From table xx, 我们可以发现我们的动态量化方法在每个数据集上相比静态的量化方案都实现了显著的压缩比提高。以全精度作为anchor，可以发现，在3种contextual 编码网络上，实现动态位宽分配后性能得到显著改善，与全精度模型相比，出现了幅度较低的bitrate increase；以静态量化作为anchor，8位和4位动态量化相比于静态方案平均码率节省分别为14.6% and 14.7%。 



图 xx 展示了3种条件编码网络在PSNR上的率失真曲线。从曲线上可以看出，8位动态量化的压缩比接近于全精度模型，4位动态量化效果也有明显提升。R-D曲线证明了所提出方法在不同分辨率和类型的视频内容上的显著性能增强。





我们进一步针对DCVC的量化性能来展开更详细的分析。

We compare the reconstruction performance of quantized ROI regions and non-ROI regions, as shown in Fig. \ref{fig4}. The figure presents the R-D curves for 32-bit baseline, 8-bit dynamic quantization, quantized ROI regions, and non-ROI regions. We observed that while the overall image quality is slightly degraded after dynamic quantization, the  ROI regions maintain performance levels close those of full precision model. In contrast, the performance in non-salient regions which are often unnoticed by the human eye, remains lower. This demonstrates that the bit-allocator precisely allocated computational resources across different areas, allocating more bits to significant foreground areas and fewer to less important background regions. In general, our proposed method achieves substantial coding gains in the ROI regions, enhancing the perceptual quality of real-time video for human eyes.

我们比较了量化的ROI区域和非ROI区域的重建性能，如图\ref{fig4}所示。图中显示了32位基线、8位动态量化、量化ROI区域和非ROI区域的R-D曲线。我们观察到，虽然动态量化后的图像质量略有下降，但ROI区域保持了接近全精度模型的性能水平。相比之下，在人眼往往无法注意到的非显著区域，其性能仍然较低。这表明该比特分配器精确地在不同的区域分配计算资源，为重要的前景区域分配更多的比特，为次要的背景区域分配更少的比特。总的来说，所提方法在ROI区域获得了可观的编码增益，提高了人眼对实时视频的感知质量。







The quantization results on standard test video datasets are shown in Tab. \ref{tab2}. We measure computational complexity using Floating Point Operations Per Second (FLOPs). Given the varying eye fixation saliency across different regions, ROI and non-ROI regions are allocated different computing resources, which facilitates efficient inference of video codecs in low-latency scenarios. In Tab. \ref{tab2}, we can clearly see that the model using 4-bit dynamic quantization achieves 89.2\% reduction in FLOPs, and with only a minimal bitrate increase compared to the 32-bit baseline. Furthermore, our method shows about 20\% decrease in FLOPs and significantly lowers frame distortion compared to the static quantization models at 4-bit and 8-bit.

在标准测试视频数据集上的量化结果如Tab所示。\ref{tab2}。我们使用每秒浮点运算次数(FLOPs)来衡量计算复杂度。考虑到不同区域的眼睛注视显著性不同，ROI和非ROI区域分配了不同的计算资源，有利于低延迟场景下视频编解码器的高效推断。在Tab中。\ref{tab2}，我们可以清楚地看到，使用4位动态量化的模型实现了89.2％的FLOPs减少，与32位基线相比，只增加了最小的比特率。此外，与静态量化模型相比，该方法在4位和8位量化时的FLOPs降低了约20％，显著降低了帧失真。





We measure computational complexity with Bit-FlOPs. Specifically, since the current CPUs are capable of performing both bit-wise XNOR and bit-count operations in parallel, the Bit-Flops of quantized convolutional layer can be calculated as $\frac{||b_x\cdot b_w||_2}{32}*$ $2*C_{in}*C_{out}*F^2*B*H*W$, which equals the Bit-OPs following [4]; $b_x,b_w$ is the bit-width of input feature $x$ and weights w.







表5中生成720p (1820 × 720)图像的操作数的位宽加权(BitOPs)。我们使用BitOPs作为计算复杂度的度量，以更好地反映位宽的减少。所提出框架在块推断上更有效，因为具有复杂结构和简单结构的局部区域是用不同的计算资源处理的。尽管相邻补丁之间的重叠区域在块推断中带来了额外的计算开销，但与基线相比，该框架实现了95.8%的BitOPs减少，与8位CARN-PAMS相比，在图像推断中实现了32.2%的BitOPs减少。GPU延迟是在支持4/8位加速的张量内核的NVIDIA Tesla T4 GPU上测量的。由于T4不支持6位的硬件加速，我们将6位赋值转换为8位。平均而言，CADyQ将Test4K图像的推理延迟提高到206.5ms，相比之下，8位量化(PAMS[31])的推理延迟为240.0ms, 32位量化的推理延迟为535.5ms。其他骨干模型的计算复杂度和开销的详细分析可以在补充文档的第D段中找到。

 



% Fig. \ref{fig2} and Fig. \ref{fig3} present the rate-distortion curves of our method, with distortion Fig. \ref{fig2} measured by PSNR and in Fig. \ref{fig3} by the MS-SSIM metric. Observations from the figures indicate that when using the static LSQ quantization method leads to performance degradation in both the 8-bit and 4-bit models, more noticeably in the 4-bit model. Upon implementing dynamic bit-width allocation in the 4-bit and 8-bit models, significant improvements are observed. For instance, on the HEVC-E dataset at a bitrate of 0.034 bits per pixel (bpp), the 4-bit dynamic model achieves a 1.4 dB gain compared to the static model. Additionally, With the same reconstruction quality, in the 1080p datasets (MCL-JCV, UVG, HEVC-B), the dynamic 4-bit/8-bit methods achieve bitrate savings of 18.2\%, 17.9\%, 19.4\%/12.3\%, 11.8\%, 13.6\% respectively, when compared to static quantization. On other resolution datasets (HEVC-C, HEVC-D, HEVC-E), dynamic quantization in 4-bit/8-bit also shows improvements, with bitrates saving of 9.5\%, 11.2\%, 12.4\%/8.2\%, 10.4\%, 11.8\% respectively. The R-D curves demonstrate significant performance enhancements of our proposed method on different resolutions and types of video content.





To explore the quantization strategy employed by bit-allocator, we visualize the bit-width assigned to the input frame of each video clip. As shown in Fig. \ref{fig5}. The Bit-allocator accurately allocates bit-widths based on the content of the video. Specifically, complex images, portraits or scenes with significant motion receive a higher bit-width for improved reconstruction quality. Conversely, regions characterized by minimal optical flow variation or simpler structures are allocated a lower bit-width to conserve computing resources. This targeted allocation helps retain more resources in high-complexity areas, effectively minimizing the exacerbation of frame distortion due to cumulative quantization errors.



如图所示，展示HEVC_B中的 "Kimono" 和 "BasketballDrive"帧的位宽

为了探索比特分配器所采用的量化策略，将每个视频片段输入帧的比特宽度可视化。如图\ref{fig5}所示。比特分配器根据视频内容准确分配比特宽度。具体来说，复杂的图像、肖像或具有显著运动的场景接收更高的位宽以提高重建质量。相反，对于光流变化最小或结构较简单的区域，则分配较低的位宽以节省计算资源。这种有针对性的分配有助于在高复杂度区域保留更多的资源，有效地减少因累积量化误差而导致的帧失真加剧。



受到基于显著性进行高效编码方法的启发



% where $\sigma(\boldsymbol{x})$ and $\sigma(\boldsymbol{v})$ represent the region features of the image and the standard deviation of the predicted optical flow, respectively. $G(\boldsymbol{x})$ and $G(\boldsymbol{v})$ represent the gradients of region features and optical flow vectors in horizontal and vertical directions. These several metrics evaluate the texture structure complexity and motion complexity of the image, and regions with complex structures or high motion content will be assigned higher bit-width. We concat these indicators into the fully connected layer $fc$, and obtain the output logits $\pi^1,\pi^2,... ,\pi^k$, corresponding to K bit-width candidates. The quantized bit-width with the largest allocation probability is selected by:





\begin{table}[]
\centering
\caption{Complexity analysis and decoding speed of DCVC-DC, DCVC-HEM and DCVC. }\label{tab4}
\begin{tabular}{cccccccc}
\hline
\multirow{2}{*}{Model} & \multirow{2}{*}{Size(MB)} & \multirow{2}{*}{Bit-Flops(G)} & \multicolumn{4}{c}{FPS}    & \multirow{2}{*}{BD-Rate(\%)} \\
                       &                           &                            & 1080p & 720p & 480p & 360p &                              \\
\hline                       
32-bit                 & 52.4                      & 2225                       & 1.3   & 5.1  & 11.8 & 24.7 & 0.0                          \\
8-bit static           & 14.4                      & 542                        & 2.9   & 11.9 & 24.5 & 41.2 & 18.1                         \\
8-bit dynamic          & 13.6                      & 462                        & 3.4   & 13.5 & 29.8 & 49.4 & 5.8                          \\
4-bit static           & 8.3                       & 282                        & 3.8   & 15.8 & 36.3 & 57.2 & 26.4                         \\
4-bit dynamic          & 7.8                       & 241                        & 4.4   & 18.2 & 41.1 & 62.2 & 12.5                \\
\hline
\end{tabular}
\end{table}



可以发现动态量化后的解码速度有了平均 2.4倍 的提升；

AdaQuant \cite{hubara2020improving} proposed a post-training quantization method based on layerwise calibration and integer programming, which achieved ultra-low performance loss on 4-bit quantization.
