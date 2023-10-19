# AE&VAE



自编码器最初提出的目的是为了学习到数据的主要规律，进行数据压缩，或特征提取，是一类无监督学习的算法。使用机器学习或深度学习手段令算法自己求解出数据表示结果的领域被称之为表征学习（Representation Learning）。而自编码器就是表征学习中的一类算法，但是后续各种改进的算法，使得自编码器具有生成能力，例如变分自编码器（VAE）的图像生成能力。

## 自编码器

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/0*fWQTibN12kZFvG3D.png" alt="img" style="zoom:50%;" />

自编码器是一种无监督学习算法, 它的目标是学习一个将输入数据编码和解码的表示。自编码器由两部分组成: 编码器和解码器。

1. 编码器: 它将输入数据 $x$ 转换为一个隐藏的表示 $z$ 。
2. 解码器: 它将隐藏的表示 $z$ 转换回与原始输入数据相似的数据。

数学上, 我们可以表示编码器和解码器为两个函数: $f$ (编码器) 和 $g$ (解码器) , 其中:

$$
\begin{aligned}z&=f(x)\\\hat{x}&=g(z)\end{aligned}
$$

自编码器的目标是最小化重构误差, 即输入$x$ 和重构的输出 $\hat{x}$ 之间的差异。常用的损失函数是均方误差(MSE):
$$
L(x,\hat{x})=||x-\hat{x}||^2
$$
**AE的局限性**

1. **潜在空间的不连续性**：从重构目标函数来看，AE关心训练数据的每个具体点如何被编码和解码，而不是整个潜在空间的结构；并且**没有潜在空间的正则化**，在基本的 AE 中，除了重构误差之外，并没有其他的正则化项来鼓励潜在空间具有某种特定结构。这与VAE不同，VAE 的 KL 散度项鼓励潜在空间接近标准正态分布，这有助于保持空间的连续性和平滑性。
2. **过度拟合**：由于 AE 的主要目标是减少重构误差，它们可能过度拟合训练数据。这意味着 AE 可能只能复制训练数据，并不能很好地生成新颖的内容。
3. **没有明确的概率模型**：与 VAE 不同，传统的 AE 不为数据或潜在变量定义明确的概率模型。这使得它们在生成新数据时缺乏理论支持。
4. **难以控制的生成**：在 AE 的潜在空间中，各个维度的含义可能是不明确的，这使得难以控制或引导生成过程。

## 变分自编码器

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/vae4.png" alt="img" style="zoom:40%;" />

变分自编码器 (Variational Autoencoders, VAE) 是自编码器的一种扩展, 它不仅可以学习数据的表示, 还可以生成新的数据。

VAE 的关键思想是引入一个概率分布来表示数据的潜在空间。与普通的自编码器不同, VAE的编码器输出的是潜在变量的均值和方差, 而不是一个固定的点。给定输入$x$, VAE 的编码器输出均值 $\mu$ 和方差 $\sigma^2$。然后, 我们从这个分布中采样潜在变量 $z$。具体地, 使用**重参数化**技巧来采样，即：
$$
z=\mu+\sigma\odot\epsilon 
$$

其中 $\epsilon$ 是从标准正态分布 $\mathcal{N}(0,1)$ 中采样的。VAE 的损失函数包含两部分：

**重构损失**: 与普通的自编码器相似, 我们希望输入和重构的输出尽可能接近，采用均方误差。

**KL 散度损失**: 这部分损失确保了潜在空间的分布接近于标准正态分布。公式为:
$$
\mathrm{KL}(q(z|x)||p(z))=-\frac12\sum_{i=1}^d(1+\log(\sigma_i^2)-\mu_i^2-\sigma_i^2)
$$

其中$d$ 是潜在空间的维度。总的损失函数是这两部分损失的和。

## 数学推导

### Overview

一般而言, 我们遇到的隐性变量模型 (如图1) 都是希望在给定数据 $\color{red}{y\in\mathbb{R}^m}$ 中找到一个隐性空间(latent space) $z\in\mathbb{R}^r$ ,其中, $r\ll m$ , 这个过程在机器学习中通常被称为降维 (dimensionality reduction)。

如果用概率图模型来描述这类隐性变量模型, 则会由两部分组成, 即生成模型 (generative model) 和推断模型 (inference model), 生成模型是指条件概率分布 $\color{red}{p(y\mid z)}$ (给定先验分布 $p(z)$)，推断模型则是指条件概率分布 $q(\color{red}{z\mid y})$ ,需要说明的是, 在这里, 将推断模型写成 $q(\cdot)$ 的形式完全是为了跟$p(\boldsymbol{z}\mid\boldsymbol{y})$ 区分开来, 避免符号混淆。

在贝叶斯模型中, 最终目标是求解出推断模型 $q(\boldsymbol{z}\mid\boldsymbol{y})$ ,而在变分自编码器中, 生成模型和推断模型都是用神经网络完成的。除了生成模型和推断模型所对应的分布, 还需要介绍一个非常重要分布, 它叫边缘似然 (marginal likelihood)：
$$
p(y)=\int p(\boldsymbol{y},\boldsymbol{z})d\boldsymbol{z}=\int p(\boldsymbol{y}\mid\boldsymbol{z})p(\boldsymbol{z})d\boldsymbol{z}
$$
虽然这条积分表达式看起来比较简洁,但想直接求解它却并不容易, 由此, 我们要先看看变分推断的解决策略。

![](https://pic2.zhimg.com/80/v2-d1ce7d4e054111b155ca0fb1be9a4341_1440w.webp)

> 假设有 $m$ 个独立的采样值, 左图显示的（encoder）推断模型 $q(z\mid y)$ 实际上就是一个降维过程, 右图展示了（decoder）生成模型 $p(y\mid z)$ , 结合先验分布 $p(z)$ 可进行后验采样。需要注意说明的是, 在变分自编码器的框架下, 推断模型中的编码 (encoding) 和生成模型中的解码 (decoder) 都受参数 $w\in\mathbb{R}^r$ 控制, 并且由神经网络训练得到。

### 变分推断

在贝叶斯学习的诸多模型中, 最为核心的就是在已知先验分布和似然的情况下找到后验分布, 实际上，变分推断作为贝叶斯推断的一种方法, 是希望尽可能地用 $q(z)$ 去逼近真实的后验分布
 $p(\boldsymbol{z}\mid\boldsymbol{y})$ , 即相应的优化问题为：
$$
q(\boldsymbol{z})=\arg\min_{q(\boldsymbol{z})}\mathrm{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}\mid\boldsymbol{y}))
$$
 其中, 符号 KL$( \cdot ) $ 全称是Kullback-Leibler divergence或者KL散度, 用于度量概率分布之间的距
 离, 它的标准定义如下:

> 【定义】对于两个给定的概率分布$q(z)$ 和$p(\boldsymbol{z}\mid\boldsymbol{y})$, 它们之间的KL散度为KL$( q( \boldsymbol{z}) \mid p( \boldsymbol{z}\mid \boldsymbol{y}) ) = \mathbb{E} _{q( \boldsymbol{z}) }\left [ \ln \frac {q( \boldsymbol{z}) }{p( \boldsymbol{z}|\boldsymbol{y}) }\right ] $
>  其中, 带有下标 $q(z)$ 的 $\mathbb{E}_{q(z)}$ 表示关于概率分布 $q(z)$ 的数学期望 。

现在, 尽管我们知道了KL散度的定义, 但它可能还是有点抽象, 尤其当我们想要求解上述优化问题的时候, 很可能无从下手。将定义中的KL散度尝试进行化简, 则
$$
\operatorname{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}\mid\boldsymbol{y}))\\
=\mathbb{E}_{q(z)}\left[\ln q(\boldsymbol{z})\right]-\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{z}\mid\boldsymbol{y})\right]\\
=\mathbb{E}_{\boldsymbol{q}(\boldsymbol{z})}\left[\ln q(\boldsymbol{z})\right]-\mathbb{E}_{\boldsymbol{q}(\boldsymbol{z})}\left[\ln p(\boldsymbol{y},\boldsymbol{z})\right]+\ln p(\boldsymbol{y})
$$
 再将等式两边稍加调整, 便可以写成如下形式
$$
\begin{aligned}
&\ln p(\boldsymbol{y})=\mathrm{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}\mid\boldsymbol{y}))-\mathbb{E}_{q(\boldsymbol{z})}\left[\ln q(\boldsymbol{z})\right] \\
&+\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{y},\boldsymbol{z})\right] \\
&\geq-\mathbb{E}_{q(\boldsymbol{z})}\left[\ln q(\boldsymbol{z})\right]+\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{y},\boldsymbol{z})\right]
\end{aligned}
$$

考虑到 $p(\boldsymbol{y},\boldsymbol{z})=p(\boldsymbol{z})p(\boldsymbol{y}\mid\boldsymbol{z})$ , 可以继续把上面的表达式化简为

$$
\begin{aligned}&=-\mathbb{E}_{q(z)}\left[\ln q(z)\right]+\mathbb{E}_{q(z)}\left[\ln p(z)\right]+\mathbb{E}_{q(z)}\left[\ln p(y\mid z)\right]\\\\&=-\mathrm{KL}(q(z)\mid p(z))+\mathbb{E}_{q(z)}\left[\ln p(y\mid z)\right]\end{aligned}
$$

 在这里, $\color{red}{\ln p(y)}$ 实际上存在一个下界 (lower bound), 即变分下界 (variational lower bound,也
 被称为Evidence lower bound,相应地, 在很多文章中简称ELBO) 为

$$
-\mathrm{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}))+\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{y}\mid\boldsymbol{z})\right]
$$
尽管我们不能直接对边缘似然$\color{red}{p(y)}$ 最大化问题进行求解, 但只要使得这里变分下界最大化, 我们同样能够得用 $q(z)$ 去逼近 $\color{red}{p(z\mid y)}.$ 在变分下界的表达式中, $q(z)$ 实际上是依赖于 $\color{red}{y}$ 的条件概率, 即 $q(z\mid y)$ , 并且变分下界组成了一个自编码器 (auto-encoder).
