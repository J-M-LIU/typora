# CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution



## Motivation

将图像分解为固定大小的patch，根据图像的局部content自适应地为patch和layer分配最优bit-width，以实现更少的精度损失。patch-layer wise quantization.

引入一个可训练的bit selector，以确定每一layer和图像patch的bit-width和量化精度。Bit selector由**量化敏感度**控制，量化敏感度是通过使用patch的图像梯度的平均幅度和layer的输入特征的标准偏差来估计的。如下图，对具有更多结构和轮廓信息的features分配更高的bit-width。

![image-20230724122042410](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230724122042410.png)

**量化敏感度**

![image-20230724175318598](https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230724175318598.png)

通过比较量化前后重建image和gt HR之间的MSE的变化，具有更复杂的结构/纹理信息的patches具有更大的梯度，同时具有更大的标准差，在量化后MSE loss 会变化更显著。

**正则化惩罚项**

在计算复杂度和重建性能中找到更好的平衡。若具有高/低量化敏感度的特征选择了低/高bit-width，需要惩罚项发挥作用，来为图像重建性能保留更多计算资源。



## Methodology

### 动态激活量化

具体激活量化公式如下：
$$
Q_{b_{i, j}^{k}}\left(\boldsymbol{x}_{i}^{j}\right)=\left\lfloor\operatorname{clamp}\left(\boldsymbol{x}_{i}^{j}, a_{k}\right) \cdot \frac{s\left(b_{i, j}^{k}\right)}{a_{k}}\right\rceil \cdot \frac{a_{k}}{s\left(b_{i, j}^{k}\right)},
$$
有K个候选量化functions即K个可选的bit-width，$b^k_{i,j}$ 为第k个量化方程中，对第 i-th patch 和第 j-th layer的bit-width。$s(b^k) = 2^{b^k-1}-1$，$a_k$ 为可学习参数，采用线性对称量化方式。

### 如何选择patch和layer的bit-width

使用一个轻量级的Bit-selector，为每个bit-wdith分配一个概率。然后，选择概率最高的bit-width:

$$
b_{i, j}^{k^{*}}=\left\{\begin{array}{ll}
\arg \max _{b_{i, j}^{k}} P_{b_{i, j}^{k}}\left(\boldsymbol{x}_{i}^{j}\right) & \text { forward } \\
\sum_{k=1}^{K} b_{i, j}^{k} \cdot P_{b_{i, j}^{k}}\left(\boldsymbol{x}_{i}^{j}\right) & \text { backward }
\end{array}\right.
$$
$P_{b_{i, j}^{k}}$ 是分配给bit-width $b_{i, j}^k$ 的概率，
$$
\sum_K P_{b_{i, j}^{k}} = 1
$$
**如何衡量**

Bit selector给具有更高量化敏感度的features，对更高的bit-width预测更大的概率，第K个候选bit-width的预测概率公式定义如下：
$$
P_{b_{i, j}^{k}}\left(\boldsymbol{x}_{i}^{j}\right)=\frac{\exp \left(f\left(\sigma\left(\boldsymbol{x}_{i}^{j}\right),\left|\nabla I_{i}\right|\right)\right)}{\sum_{k=1}^{K} \exp \left(f\left(\sigma\left(\boldsymbol{x}_{i}^{j}\right),\left|\nabla I_{i}\right|\right)\right)},
$$
**Backpropagation**

选择最大概率的量化函数是一个离散不可微的过程，不能进行端到端的优化。因此，采用STE使过程可微。在反向传播过程中，离散bit-width被其**可微近似**取代，其中每个候选位宽由位bit selector预测的概率分布加权（上一公式） 
