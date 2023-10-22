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
2. **过拟合**：由于 AE 的主要目标是减少重构误差，它们可能过度拟合训练数据。这意味着 AE 可能只能复制训练数据，并不能很好地生成新颖的内容。
3. **没有明确的概率模型**：与 VAE 不同，传统的 AE 不为数据或潜在变量定义明确的概率模型。这使得它们在生成新数据时缺乏理论支持。
4. **难以控制的生成**：在 AE 的潜在空间中，各个维度的含义可能是不明确的，这使得难以控制或引导生成过程。

## 变分自编码器

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/vae4.png" alt="img" style="zoom:40%;" />

变分自编码器 (Variational Autoencoders, VAE) 是自编码器的一种扩展, 它不仅可以学习数据的表示, 还可以生成新的数据。

> AE中使用单个值来描述输入图像在潜在特征上的表现。但在实际情况中，更倾向于将每个潜在特征表示为可能值的范围。例如，如果输入蒙娜丽莎的照片，将微笑特征设定为特定的单值（相当于断定蒙娜丽莎笑了或者没笑）显然不如将微笑特征设定为某个取值范围（例如将微笑特征设定为x到y范围内的某个数，这个范围内既有数值可以表示蒙娜丽莎笑了又有数值可以表示蒙娜丽莎没笑）更合适。而变分自编码器便是用“取值的概率分布”代替原先的单值来描述对特征的观察的模型，经过变分自编码器的编码，每张图片的微笑特征不再是自编码器中的单值而是一个概率分布。

VAE 的关键思想是引入一个概率分布来表示数据的潜在空间。与普通的自编码器不同, VAE的编码器输出的是潜在变量的均值和方差, 而不是一个固定的点。给定输入$x$, VAE 的编码器输出均值 $\mu$ 和方差 $\sigma^2$。然后, 我们从这个分布中采样潜在变量 $z$。具体地, 使用**重参数化**技巧来采样，即：
$$
z=\mu+\sigma\odot\epsilon 
$$

其中 $\epsilon$ 是从标准正态分布 $\mathcal{N}(0,1)$ 中采样的。VAE 的损失函数包含两部分：

**重构损失**: 与普通的自编码器相似, 我们希望输入和重构的输出尽可能接近，采用均方误差。

**KL 散度损失**: 这部分损失确保了潜在空间的分布接近于标准正态分布：
$$
loss = ||x-\hat{x}||^2 + KL(q(\boldsymbol{z})\mid p(\boldsymbol{z}\mid\boldsymbol{y}))
$$
其中$d$ 是潜在空间的维度。总的损失函数是这两部分损失的和。

**训练流程**

- 首先，将输入编码为在隐空间上的分布；
- 第二，从该分布中采样隐空间中的一个点；
- 第三，对采样点进行解码并计算出重建误差；
- 最后，重建误差通过网络反向传播。

## 数学推导

### Overview

一般而言, 我们遇到的隐性变量模型 (如图1) 都是希望在给定数据 $\color{red}{y\in\mathbb{R}^m}$ 中找到一个隐性空间(latent space) $z\in\mathbb{R}^r$ ,其中, $r\ll m$ , 这个过程在机器学习中通常被称为降维 (dimensionality reduction)。

如果用概率图模型来描述这类隐性变量模型, 则会由两部分组成, 即生成模型 (generative model) 和推断模型 (inference model), 生成模型是指条件概率分布 $\color{red}{p(y\mid z)}$ (给定先验分布 $p(z)$)，推断模型则是指条件概率分布 $\color{red}q({z\mid y})$ ,需要说明的是, 在这里, 将推断模型写成 $q(\cdot)$ 的形式完全是为了跟$p(\boldsymbol{z}\mid\boldsymbol{y})$ 区分开来, 避免符号混淆。

在贝叶斯模型中, 最终目标是求解出推断模型 $q(\boldsymbol{z}\mid\boldsymbol{y})$ ,而在变分自编码器中, 生成模型和推断模型都是用神经网络完成的。而这个后验分布 $q(\boldsymbol{z}\mid\boldsymbol{y})$ 很难直接求解，因为涉及到如下的边缘似然 (marginal likelihood)计算，很难进行计算。
$$
p(y)=\int p(\boldsymbol{y},\boldsymbol{z})d\boldsymbol{z}=\int p(\boldsymbol{y}\mid\boldsymbol{z})p(\boldsymbol{z})d\boldsymbol{z}
$$
虽然这条积分表达式看起来比较简洁，但想直接求解它却并不容易, 由此, 我们要先看看变分推断的解决策略。

![](https://pic2.zhimg.com/80/v2-d1ce7d4e054111b155ca0fb1be9a4341_1440w.webp)

> 假设有 $m$ 个独立的采样值, 左图显示的（encoder）推断模型 $q(z\mid y)$ 实际上就是一个降维过程, 右图展示了（decoder）生成模型 $p(y\mid z)$ , 结合先验分布 $p(z)$ 可进行后验采样。需要注意说明的是, 在变分自编码器的框架下, 推断模型中的编码 (encoding) 和生成模型中的解码(decoder) 都受参数 $w\in\mathbb{R}^r$ 控制, 并且由神经网络训练得到。

### 变分推断 **Variational inference**

在贝叶斯学习的诸多模型中, 最为核心的就是在已知先验分布和似然的情况下找到后验分布, 实际上，变分推断作为贝叶斯推断的一种方法, 是希望尽可能地用 $q(z)$ 去逼近真实的后验分布 $p(\boldsymbol{z}\mid\boldsymbol{y})$ , 即相应的优化问题为：
$$
q(\boldsymbol{z})=\arg\min_{q(\boldsymbol{z})}\mathrm{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}\mid\boldsymbol{y}))
$$
其中, 符号 KL$( \cdot ) $ 全称是Kullback-Leibler divergence或者KL散度, 用于度量概率分布之间的距离, 它的标准定义如下:

> 【定义】对于两个给定的概率分布$q(z)$ 和$p(\boldsymbol{z}\mid\boldsymbol{y})$, 它们之间的KL散度为
>
> $KL( q( \boldsymbol{z}) \mid p( \boldsymbol{z}\mid \boldsymbol{y}) ) = \mathbb{E} _{q( \boldsymbol{z}) }\left [ \ln \frac {q( \boldsymbol{z}) }{p( \boldsymbol{z}|\boldsymbol{y}) }\right ] $
> 其中, 带有下标 $q(z)$ 的 $\mathbb{E}_{q(z)}$ 表示关于概率分布 $q(z)$ 的数学期望 。

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

在这里, $\color{red}{\ln p(y)}$ 实际上存在一个下界 (lower bound), 即变分下界 (variational lower bound,也被称为Evidence lower bound,相应地, 在很多文章中简称ELBO) 为

$$
-\mathrm{KL}(q(\boldsymbol{z})\mid p(\boldsymbol{z}))+\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{y}\mid\boldsymbol{z})\right]
$$
尽管我们不能直接对边缘似然 $\color{red}{p(y)}$ 最大化问题进行求解, 但只要使得这里变分下界最大化, 我们同样能够得用 $q(z)$ 去逼近 $\color{red}{p(z\mid y)}.$ 在变分下界的表达式中, $q(z)$ 实际上是依赖于 $\color{red}{y}$ 的条件概率, 即 $q(z\mid y)$ , 并且变分下界组成了一个自编码器 (auto-encoder).



### 高斯变分自编码器 **Gaussian variational auto-encoder**

在变分自编码器中, 推断模型 $q(z\mid y)$ 很明显是通过神经网络来完成的, 以高斯变分自编码器为例, 假设概率分布 $p(z)$ 和 $q(z\mid y)$ 都服从高斯分布 (正态分布), 具体形式为

$$
q(\boldsymbol{z}\mid\boldsymbol{y})=\mathcal{N}(\boldsymbol{\mu}(\boldsymbol{y},\boldsymbol{w}),\mathrm{diag}(\boldsymbol{\sigma}^2(\boldsymbol{y},\boldsymbol{w})))\\
p(\boldsymbol{z})=\mathcal{N}(\mathbf{0},I)
$$
其中, 参数 $\color{red}{w}$ 表示神经网络的权重 (weight), 用神经网络优化 $\color{red}{w}$ 之后, 均值 $\mu(y,w)\in\mathbb{R}^r$ 和协方差矩阵对角线的元素 $\sigma^2(y,\boldsymbol{w})\in\mathbb{R}^r$ 就能够预测出来了。在上述假设中, 符号 $\mathcal{N}(\cdot)$ 是高斯分布的简写, 取自normal distribution (正态分布) 的首字母; $I\in\mathbb{R}^{r\times r}$ 表示单位矩阵。

回想前面提到的变分下界, 表达式为
$$
-\mathrm{KL}(q(\boldsymbol{z}\mid\boldsymbol{y})\mid p(\boldsymbol{z}))+\mathbb{E}_{q(\boldsymbol{z})}\left[\ln p(\boldsymbol{y}\mid\boldsymbol{z})\right]
$$
一般而言, 尽管概率分布 $q(z\mid y)$ 和 $p(z)$ 之间的KL散度能够直接计算出来, 然而, 在前面给出的高斯分布假设中, 两个概率分布都隐藏着参数 $\color{red}{w}$ 。

为了使得优化目标与参数估计能够和神经网络的工作机制契合, 一个经典的处理方法是 **reparameterization trick**, 这个技巧的核心是变换 (transformation, $g(\cdot))$, 即在服从标准正态分布的随机变量 $\epsilon_i\sim\mathcal{N}(0,1),i=1,2,\ldots,r$ 基础上对随机变量 $z\sim q(z\mid y)$ 进行重写, 形式如下：
$$
z_i=g(\boldsymbol{y},\epsilon_i)=\mu_i(\boldsymbol{y})+\epsilon_i\sigma_i(\boldsymbol{y})
$$
总体而言, 给定数据 $\color{red}{y}\in\mathbb{R}^m$ ，需要优化 (最小化)的目标函数 (loss function,损失函数)为：
$$
\mathcal{L}_{\mathrm{VAE}}(\boldsymbol{w})=\mathrm{KL}(q(\boldsymbol{z}\mid\boldsymbol{y})\mid p(\boldsymbol{z}))-\frac1n\sum_{k=1}^n\ln p(\boldsymbol{y}\mid\boldsymbol{z}_k)
$$
其中, $\color{red}{z_k=g(y,\epsilon_k)}$，随机变量 $\epsilon_k\sim\mathcal{N}(0,I)$ 采样自标准的多元正态分布，$\color{red}{n}$ 表示随机采样的次数, 方便起见, 可以取 $n=1$ 。

> 若已知神经网络的权重 $\boldsymbol{w}$ ：
> (1)解码器为: 对随机变量 $z$ 进行采样, 取自标准的多元正态分布 $p(z)=N(0,I)$ ,然后对随机变量 $\boldsymbol{y}\sim p(\boldsymbol{y}\mid\boldsymbol{z})$ 进行采样；
> (2)编码器为: 对随机变量$z$ 进行采样, 取自 $z\sim p(z\mid y)$ ,其中, 先完成 $\epsilon\sim\mathcal{N}(0,I)$ 的采样过程,然后采用变换 $z=g(y,\epsilon)$ .



### 重参数技巧

重参数化技巧是变分自编码器(VAE) 中的一个关键技巧, 它主要解决了以下问题: 如何在一个包含随机性的模型中进行梯度下降优化?

为了理解这一点, 让我们考虑以下问题：

在 VAE 中, 我们希望从一个参数化的分布 $q(z|x)$ 中采样潜在变量 $z$。具体来说, 给定输入 $x$ , 编码器输出均值 $\mu$ 和方差 $\sigma^2$, 并希望从此分布中采样 $z$。一个直观的方法是直接从 $\mathcal{N}(\mu,\sigma^2)$ 中采样 $z$, 但这样做的问题是, 采样操作是随机的, 不可微分, 因此不能直接用于基于梯度的优化。

为了解决这个问题, 我们引入了重参数化技巧。核心思想是将随机性与模型参数分开。具体来说, 我们不直接从$N(\mu,\sigma^2)$ 中采样 $z$, 而是从标准正态分布 $N(0,1)$ 中采样一个噪声 $\epsilon$ , 然后通过以下转换获得 $z:$
$$
z=\mu+\sigma\odot\epsilon 
$$

在这个方案中, 随机性仅来自$\epsilon$,而$z$ 是$\mu$、$\sigma$ 和$\epsilon$的确定性函数。这意味着我们可以相对于$\mu$ 和 $\sigma$ 计算 $z$ 的梯度, 从而使得基于梯度的优化成为可能。 

总之, 重参数化技巧的主要目的是在引入随机性的同时, 还能够进行梯度下降优化。
