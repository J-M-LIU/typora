# Space-Time Distillation for Video Super-Resolution



## Introducton

视频超分辨常见于部署在资源有限设备上，因此需要参数量更少、复杂度更低的模型。本文提出一种知识蒸馏方法，主要对空间知识和时间知识进行蒸馏：对于空间蒸馏，利用来自教师网络的高频视频内容的空间注意力图作为学生网络的训练目标，进一步用于转移空间建模能力。在时间蒸馏方面，通过提取时间记忆单元的特征相似性，缩小紧凑模型和复杂模型之间的性能差距，时间记忆单元是从使用ConvLSTM在训练片段中生成的特征图序列中编码的。

![image-20231024153355538](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231024153355538.png)

> 蒸馏框架 (a)VSR结构(如EDVR[37])。将网络产生的残差图和输入参考帧的双三次上采样结果相加，得到输出超分辨率帧$SR_t$。(b)空间蒸馏：利用来自教师网络的空间注意力图（$M_t^T$）作为从学生网络提取的空间注意力图（$M_t^S$）的蒸馏目标。(c)时间蒸馏：提取多帧特征映射，通过ConvLST编码为时间记忆单元 $C_T^{SR}$ 和 $C_S^{SR}$ ，并最小化教师网络和学生网络之间的差距。



## Space-Time Distillation

给定2k + 1个连续的LR帧 $I_{[t−k:t+k]}$，表示中间帧 $I_t$为参考帧，其他帧为相邻帧。VSR的目标是估计一个上采样的参考框架 $SR_t$，期望它接近于真实的 $HR_t$。超分辨的基本结构是：多个输入帧通过精心设计的运动对齐模块，每个输入相邻帧在 feature domain 与参考帧对齐，在 spatial-tempora domain 进行聚合。$F_t^{LR}$ 表示对齐(and)/融合(or)的特征，并通过 pixelshuffel 将其送入重建主干，以估计与 $HR_t$ 具有相同空间分辨率的 $F_t^{SR}$ 。最后通过对 $F_t^{SR}$ 卷积来减少通道数得到重建 $SR_t$。**本文使用特征 $F_t^{SR}$ 而非 $F_t^{LR}$ 来进行蒸馏，根据后续的消融实验可发现，使用 $F_t^{SR}$ 可以提高重建准确度。**

### Space Distillation（SD）

为了重建参考帧，高频细节是非常关键的。作者受到了基于激活的注意力蒸馏的启发，设计了一个SD方案来模拟空间的表征能力，它通过提取T的空间注意力图来进行。其中，$F^{SR}_T$ 和 $F^{SR}_S$ 分别代表教师网络和学生网络的特征图。文章中使用C、H和W来分别表示特征图的通道、高度和宽度。

空间注意力图的生成可以被视为找到一个映射函数 $M : ℝ^{C\times W\times H} → ℝ^{W\times H}$。这个空间注意力图包含了丰富的高频视频内容。映射函数可以定义为以下三种操作中的一种：
$$
\begin{gathered}
\mathcal{M}_{sum}(F_t^{SR})=\sum_{i=1}^C|F_{t,i}^{SR}|, \\
\mathcal{M}_{sum}^2(F_t^{SR})=\sum_{i=1}^C|F_{t,i}^{SR}|^2, \\
\mathcal{M}_{max}^{2}(F_{t}^{SR})=max_{i=1}^{C}|F_{t,i}^{SR}|^{2}, 
\end{gathered}
$$
其中 $F^{SR}_{t,i}$  是通道维度中的第i个片段。

图3显示了使用这三种映射函数提取的空间注意力图的输入参考帧和可视化结果。与 $M^2_{max}$ 相比，$M^2_{sum}$ 描述了场景的细节更为清晰和准确，因为它在全局机制中计算权重，而不仅仅是简单地选择最大值。

<img src="/Users/liujiamin/Library/Application Support/typora-user-images/image-20231024170423987.png" alt="image-20231024170423987" style="zoom:50%;" />

通过映射函数 $M^2_{sum}$ ，可以计算教师网络和学生网络的空间注意力图：
$$
M_t^{\mathrm{T}/\mathrm{S}}=\mathcal{M}_{sum}^2(F_{\mathrm{T}/\mathrm{S},t}^{SR}).
$$
将空间注意力图 $M_t^{\mathrm{S}}$ 逼近  $M_t^{\mathrm{T}}$ 来训练学生网络S。将空间注意力图中的知识从教师传递给学生，在学习视频高频细节方面更好地模仿教师。优化学生网络S的SD损失为：
$$
\mathcal{L}_{SD}=\frac1N\sum_{t=1}^N\mathcal{L}_2(M_t^\mathrm{s},M_t^\mathrm{T})
$$
N是训练片段中的帧数，采用滑动窗口方法来创建训练pairs，边界帧使用复制帧来创建对。



### Time Distillation (TD)

利用多帧图像之间的相关性是VSR的关键步骤。复杂教师网络由于设计了良好的帧对齐和融合结构，具有更强的处理大运动时间信息的能力。TD旨在将教师网络的时间建模能力迁移到学生网络。

给定 $2k + 1$ 连续的低分辨率（LR）帧 $I_{[t−k:t+k]}$，对应的学生网络和教师网络的特征映射 $F_{[t−k:t+k]}^{SR}$ 从重建 backbone 中输出（具体见图1）。将 2k + 1 $F_t^{SR}$ 按顺序输入到ConvLSTM单元[30]中。输出的短时记忆单元($C_{SR}$)通过持续地将前一个特征的隐藏状态传递到最后一个单元，来记录输入视频片段的长期时域信息。将由  ConvLSTM 单元编码的教师网络和学生网络的时间记忆单元分别表示为 $C_{SR}^T$ 和 $C_{SR}^S$。优化学生网络S的TD损失为：
$$
\mathcal{L}_{TD}=\mathcal{L}_d(C_\mathrm{T}^{SR},C_\mathrm{s}^{SR})
$$


<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20231024171936981.png" alt="image-20231024171936981" style="zoom:50%;" />

为了提取多帧时间信息，教师网络和学生网络共享ConvLSTM单元的权重。请注意，当ConvLSTM单元中的权重和偏差都等于零时，可能存在模型崩溃点。在训练过程中，当TD损失值低于1e−8时，固定ConvLSTM的参数以防止模型崩溃。

### Loss

$$
\mathcal{L}_{rec}=\sqrt{\left\|SR_t-HR_t\right\|^2+\varepsilon^2} \\
\mathcal{L}=\mathcal{L}_{rec}+\lambda_1\mathcal{L}_{SD}+\lambda_2\mathcal{L}_{TD}
$$

