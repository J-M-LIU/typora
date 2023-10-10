## Folding BatchNorm

Batch Normalization实现过程如下：
$$
\begin{array}{l}
\text { Input: Values of } x \text { over a mini-batch: } \mathcal{B}=\left\{x^{(1) \ldots (m)}\right\} \text {; } \\
\text { Parameters to be learned: } \gamma, \beta \\
\text { Output: }\left\{y^{(i)}=\mathrm{BN}_{\gamma, \beta}\left(x^{(i)}\right)\right\} \\
\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \quad \quad / / \text { mini-batch mean } \\
\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x^{(i)}-\mu_{\mathcal{B}}\right)^{2} \quad \text { // mini-batch variance } \\
\widehat{x}^{(i)} \leftarrow \frac{x^{(i)}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \quad \text { // normalize } \\
y^{(i)} \leftarrow \gamma \widehat{x}^{(i)}+\beta \equiv \mathrm{BN}_{\gamma, \beta}\left(x^{(i)}\right) \quad \text { // scale and shift } \\
\end{array}
$$
其中 $x^{(i)}$ 是网络中间某一层的激活值，$\mu_{\beta}$、$\sigma_{\beta}$ 分别是均值和方差，$y^{(i)}$ 是经过BN层的输出。

**一般卷积层与BN层合并**

Folding BatchNorm 不是量化才有的操作，在一般的网络中，为了加速网络推理，我们也可以把 BN 合并到 Conv 中。

假设有一个已经训练好的 Conv 和 BN：

<img src="https://pic1.zhimg.com/80/v2-9a01a3657d1f6453b8111fd3d1359244_1440w.webp" style="zoom:50%;" />

假设卷积层的输出为：
$$
y = \sum_i^N w_i x_i +b
$$
图中 BN 层的均值和标准差可以表示为 $\mu_y$、$\sigma_y$，根据论文的表述，BN层的输出为：
$$
\begin{aligned}
y_{b n} & =\gamma \hat{y}+\beta \\
& =\gamma \frac{y-\mu_{y}}{\sqrt{\sigma_{y}^{2}+\epsilon}}+\beta
\end{aligned}
$$
然后把 (2) 代入 (3) 中可以得到：
$$
y_{b n}=\frac{\gamma}{\sqrt{\sigma_{y}^{2}+\epsilon}}\left(\sum_{i}^{N} w_{i} x_{i}+b-\mu_{y}\right)+\beta
$$
用 $\gamma'$ 来表示 $\frac{\gamma}{\sqrt{\sigma_{y}^{2}+\epsilon}}$，那么 (4) 可以简化为：
$$
\begin{aligned}
y_{b n} & =\gamma^{\prime}\left(\sum_{i}^{N} w_{i} x_{i}+b-\mu_{y}\right)+\beta \\
& =\sum_{i}^{N} \gamma^{\prime} w_{i} x_{i}+\gamma^{\prime}\left(b-\mu_{y}\right)+\beta
\end{aligned}
$$
(5) 式形式上跟 (2) 式一模一样，因此它本质上也是一个 Conv 运算，只需要用 $w_i' = \gamma'w_i$ 和 $b'= \gamma'(b-\mu_y)+\beta$ 来作为原来卷积的 weight 和 bias，就相当于把 BN 的操作合并到了 Conv 里面。实际 inference 的时候，由于 BN 层的参数已经固定了，因此可以把 BN 层 folding 到 Conv 里面，省去 BN 层的计算开销。





