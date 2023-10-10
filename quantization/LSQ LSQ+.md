LSQ 是 IBM 在 2020 年发表的一篇文章，从题目意思也可以看出，文章是把量化参数 step size (也叫 scale) 也当作参数进行训练。这种把量化参数也进行求导训练的技巧也叫作**可微量化参数**。在这之后，高通也发表了增强版的 LSQ+，把另一个量化参数 zero point 也进行训练，从而把 LSQ 推广到非对称量化中。

**普通量化训练**

在量化训练中需要加入伪量化节点 (Fake Quantize)，这些节点做的事情就是把输入的 float 数据量化一遍后，再反量化回 float，以此来模拟量化误差，同时在反向传播的时候，发挥 STE 的功能，把导数回传到前面的层。

<img src="https://pic3.zhimg.com/80/v2-a4db8c88fa80b458f54436dda98138ee_1440w.jpg" style="zoom:50%;" />

Fake Quantize 的过程可以总结成以下公式 (为了方便讲解 LSQ，这里采用 LSQ 中的对称量化的方式)：
$$
\bar{v} = round(clip(\frac{v}{s},-Q_N,Q_P))\\
\hat{v} = \bar{v} \times S
$$
其中，$v$是float的输入，$\bar{v}$ 是量化后的数据（仍然使用float来存储，但数值由于做了round操作，因此是整数），$\hat{v}$ 是反量化的结果。$-Q_N$和 $Q_P$ 分别是量化数值的最小值和最大值（在对称量化中，$Q_N$、$Q_P$ 通常是相等的），$S$ 是量化参数。
由于 round 操作会带来误差，因此 $\hat{v}$ 和 $v$ 之间存在量化误差，这些误差反应到 loss 上会产生梯度，这样就可以反向传播进行学习。每次更新weight后，我们会得到新的float的数值范围，然后重新估计量化参数$S$，之后，开始新一次迭代训练：
$$
S = \frac{|v|_{max}}{Q_P}
$$
**LSQ**

可以看到，上面这个过程的量化参数都是根据每一轮的权重计算出来的，而整个网络在训练的过程中只会更新权重的数值。

**LSQ 想做的，就是把这里的 S 也放到网络的训练当中，而不是通过权重来计算。**

这个导数可以这样计算：把 (1)(2) 式统一得到：
$$
\begin{aligned}
\hat{v} & =\operatorname{round}\left(\operatorname{clip}\left(v / s,-Q_{N}, Q_{P}\right)\right) \times s \\
& =\left\{\begin{array}{ll}
-Q_{N} \times s & v / s<=-Q_{N} \\
\operatorname{round}(v / s) \times s & -Q_{N}<v / s<Q_{P} \\
Q_{P} \times s & v / s>=Q_{P}
\end{array}\right.
\end{aligned}
$$
然后对 $S$ 求导得到：
$$
\frac{\partial \hat{v}}{\partial s}=\left\{\begin{array}{ll}
-Q_{N} & v / s<=-Q_{N} \\
\operatorname{round}(v / s)+\frac{\partial r o u n d(v / s)}{\partial s} \times s & -Q_{N}<v / s<Q_{P} \\
Q_{P} & v / s>=Q_{P}
\end{array}\right.
$$
$round(v/s)$ 这一步的导数可以通过 STE 得到：
$$
\begin{aligned}
\frac{\partial \operatorname{round}(v / s)}{\partial s} & =\frac{\partial(v / s)}{\partial s} \\
& =-\frac{v}{s^{2}}
\end{aligned}
$$
最终得到论文中的求导公式：
$$
\frac{\partial \hat{v}}{\partial s}=\left\{\begin{array}{ll}
-Q_{N} & v / s<=-Q_{N} \\
-\frac{v}{s}+\operatorname{round}(v / s) & -Q_{N}<v / s<Q_{P} \\
Q_{P} & v / s>=Q_{P}
\end{array}\right.
$$
假设把量化范围固定在［0, 3］区间，（即$Q_N$＝0，$Q_P$＝3）下面图表示量化前的 $v$ 和反量化后的$\hat{v}$ 之间的映射关系（假设 S＝1），round采用四舍五入的原则,。在0.5这个地方，$\hat{v}$ 是会从0突变到1的，从而带来巨大的量化误差。

<img src="https://pic4.zhimg.com/80/v2-ba2753f4c017927ddc9428597a38b3e7_1440w.webp" style="zoom:67%;" />

因此，从 0.5 的左侧走到右侧，梯度应该是要陡然增大的。

在下图中，作者就对比了 QIL、PACT 和 LSQ (前面两个是另外两种可微量化参数的方法) 在这些突变处的梯度变化，结果发现，QIL 和 PACT 在突变处的梯度没有明显变化，还是按照原来的趋势走，而 LSQ 则出现了一个明显突变 (注意每条虚线右侧)。因此，LSQ 在梯度计算方面是更加合理的。

<img src="https://pic1.zhimg.com/80/v2-c33651fb2e54fed1f801166645cc701c_1440w.webp" style="zoom:67%;" />

此外，作者还认为，在计算 s 梯度的时候，还需要兼顾模型权重的梯度，二者差异不能过大，因此，作者设计了一个**比例系数**来约束 s 的梯度大小：
$$
R=\frac{\partial_{s} L}{s} / \frac{\left\|\partial_{w} L\right\|}{\|w\|} \approx 1
$$
同时，为了保持训练稳定，作者在 s 的梯度上还乘了一个**缩放系数 g**，对于 weight 来说，$g = 1/\sqrt{N_WQ_P}$ ，对于feature，$g = 1/\sqrt{N_FQ_P}$，N是weight和feature的size；然后采用 $S = \frac{2|v|}{\sqrt{Q_P}}$ 的方式初始化 S。

**LSQ+**

LSQ+ 的思路和 LSQ 基本一致，就是把零点 (zero point，也叫 offset) 也变成可微参数进行训练。加入零点后，(1)(2) 式就变成了：
$$
\begin{array}{l}
\bar{v}=\operatorname{round}\left(\operatorname{clip}\left((v-\beta) / s,-Q_{N}, Q_{P}\right)\right) \\
\hat{v}=\bar{v} \times s+\beta
\end{array}
$$
之后就是按照 LSQ 的方式分别计算导数 $\frac{\partial{\hat{v}}}{\partial{s}}$ 和 $\frac{\partial{\hat{v}}}{\partial{\beta}}$，再做量化训练。