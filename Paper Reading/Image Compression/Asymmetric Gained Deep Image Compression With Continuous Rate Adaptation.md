# Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation





## Intro

  提出了一种连续速率可调的学习图像压缩框架，即非对称增益变分自动编码器（AGVAE）。 AG-VAE 利用一对**增益单元** gain unit 在一个模型中实现**离散速率自适应**，而额外的计算可以忽略不计。并且通过使用**指数插值**，在不影响性能的情况下实现**连续速率自适应**。除此之外，部署了非对称的熵模型提高模型的性能增益。



## Proposed Method

### 目前方法的缺陷

在之前的图像压缩工作中，都是通过改变损失函数中的 $\lambda$ 超参数调整模型的码率，这会导致无法将图像压缩至固定码率点的情况，并且要为此训练多个模型，需要消耗大量的训练时间和存储模型所需要的空间。为此，单个模型能够覆盖多个码率的情形有很大的应用需求。

在以往的可变速率技术方案中，基于RNN方案进行渐进式的图像编码，但是RD性能比较差，基于条件的conditional卷积网络复杂度高并且占用内存大，可变的量化bin size方式会导致性能的下降，此外对于BottleNeck层的尺度缩放方案在低码率的情况下会掉性能。

![](https://img-blog.csdnimg.cn/ad7dc8de24cd4bb4b0bb1968e358259b.png#pic_center)

在编解码方案中，不同通道对最后的重建质量的影响是不同的。作者探索基线方案中，被量化中的前32个通道信息对最后重建图像质量的影响，得出不同通道有不同重要性的结论，并且对通道进行scale的缩放，被量化后的潜在表示值乘以尺度缩放因子，得到潜在表示缩放后的重建质量。

### 整体方案
​       整体方案还是和Google[1]的网络一致，对比框架优化了 Gain Unit 单元，扩展了自回归模型中的Mask Convolution，从1个5x5和扩展成 3x3，5x5，7x7的网络，并且文章中有优化熵模型，从单高斯模型扩展到高斯分布的两侧采用不同方差的半边高斯分布。

![](https://img-blog.csdnimg.cn/abb8e15adaae4c88a6667538562c628e.png#pic_center)

### Gain Unit

标记编码器的输出 $y\in R^{c,h,w}$ 。$y_i\in R^{h,w}$ 则表示单个通道的潜在表示，其中$i\in C$。 Gain Unit 由一个增益矩阵 $M\in R^{c,n}$ 组成，表示这个矩阵实际上是为每一个通道的 latent representation $y_i$ 分配一个长度为 n 的向量，$ m_s = \{ m_{(s,0)},m_{(s,1)},m_{(s,2)},...m_{(s,c-1)} \}$。即每个$m_{s,i}$ 是一个长度为n的向量，对于每个通道的操作表示如下：$\overline{y}=y_i\times m_{s,i}$ ，简而言之就是每一个通道上的潜在表示都会乘以对应向量中的某个值。



### Discrete Variable Rate with Gain Units 离散可变码率

编码器的输出 $y$ 经过 Unit Gain 单元进行处理缩放之后，得到 $\overline{y}$ ，之后得到量化后的潜在表示$\hat{y}=round(\overline{y})$，解码端同样部署 Inverse-Unit Gain，从熵解码器中得到 $\hat{y}$ ,然后进行对应拟变换得到 $y^{^{\prime}}=InverseGain(\hat{y}_i\times m_{s,i}^{^{\prime}})$​​。

整体框架的损失函数优化主流的损失函数基本保持一致：
$$
\sum_{\theta,\phi,\varphi,\psi}^{n-1}R_{\varphi}\left(Q\left(G_{\psi}\left(f_{\theta}\left(x\right),s\right)\right)\right)+\beta_{s}\cdot D\left(x,g_{\phi}\left(IG_{\tau}\left(Q\left(G\left(f_{\theta}\left(x\right),s\right)\right),s\right)\right)\right)
$$
其中，$R_\varphi$ 项表示码率，$D$表示失真，$\beta_s$ 表示训练模型中，失真和码率的权衡，$\beta_\mathrm{s}$ 越大，则表示模型越注重重建图像的质量。$\beta_s$ 从一组预定义好的Lagrange参数集中选取，$\beta_s\in B, B \in \R^n$ 。每一对增益向量 $\{m_s, m_s'\}$ 都对应于一个预定义的 $\beta _s$。（n个对应关系，相当于R-D曲线上有n个预定义节点？）

在推理阶段，可以训练的Unit Gain 矩阵中获取到有映射关系的$m_s,m_s^{^{\prime}}$对潜在表示$y$ 和 $\hat{y}$ 进行缩放，得到对应几个离散情况下的离散点，如下图所示，训练了基于 mse loss 和 (1-msssim) loss 的两个模型。并且通过修改Gain Unit矩阵中的对应 $m_s,m_s^{^{\prime}}$ 向量，得到的离散RD曲线，离散的模型记为DVR模型。

### Exponential Interpolation 指数插值实现连续可变码率

使用不同的增益单元对 $m_s,m_s^{\prime}$ 来实现离散单模型多码率，连续速率自适应可以通过增益单元之间的插值来实现。可以对 $m_s,m_s^{\prime}$ 和 $m_{s-1}^{\prime},m_{s-1}^{\prime}$ 进行差值完成连续可变速率的实现。为了确保不同的 $m_s,m_s^{^{\prime}}$ 之间的对于潜在表示 $y$ 和 $\hat{y}$ 的缩放结果是一致的，对不同的$m_s,m_s^{^{\prime}}$有以下约束
$$
m_s*m_s^{'}=m_t*m_t^{'}=C
$$
$m_s,m_s^{^{\prime}}$和$m_t, m_t^{^{\prime }}$ $( r, t\in [ 0, 1, . . . n- 1]$)表示不同的增益矢量单元对应在不同的$\beta_s$ 和 $\beta_t$。有以下公式：
$$\left(m_{r}\cdot m_{r}^{'}\right)^{l}\cdot\left(m_{t}\cdot m_{t}^{'}\right)^{1-l}=C,\\\left[\left(m_{r}\right)^{l}\cdot\left(m_{t}\right)^{1-l}\right]\cdot\left[\left(m_{r}^{'}\right)^{l}\cdot\left(m_{t}^{'}\right)^{1-l}\right]=C,\\m_{v}=\left[\left(m_{r}\right)^{l}\cdot\left(m_{t}\right)^{1-l}\right],m_{v}^{'}=\left[\left(m_{r}^{'}\right)^{l}\cdot\left(m_{t}^{'}\right)^{1-l}\right],$$
此处 $m_v,m_v^{\prime}$ 表示 $m_r,m_r^{\prime}$ 和 $m_tm_t^{\prime}$ 之间的差值系数，通过控制参数$l$来表示 $m_v,m_v^{^{\prime}}$ 的取值情况，当$l$从0取到1时，模型能够取到
两个离散点$m_r,m_r^{^{\prime}}$ 和 $m_tm_t^{^{\prime}}$​之间所有的连续的码率点，从而实现连续可变速率的目的。结果如下图所示：