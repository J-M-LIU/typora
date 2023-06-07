# Image Compression based on Deep Learning

​	大多数基于深度框架的图像压缩都是有损压缩

<img src="https://pic1.zhimg.com/80/v2-a004a6ca24421dd441134324362c57cc_1440w.jpg" style="zoom:33%;" />



<img src="https://pic2.zhimg.com/80/v2-c411bfca55bec3f85591ab9b866598e5_1440w.webp" style="zoom:30%;" />

- 预测编码/残差编码
- 统计编码/熵编码
- 变换编码
- 混合编码



**传统图像压缩**

变换编码——>量化——>熵编码





The entropy model is a critical component in learned image compression, which predicts the probability distribution of the quantized latent representation in the encoding and decoding modules. Different entropy models have different ways of capturing the correlations and dependencies in the latent representation. Here are some differences between the entropy models you mentioned:

- [The **factorized model** assumes that each element of the latent representation is independent and identically distributed (i.i.d.), and uses a fixed parametric distribution (such as a generalized divisive normalization) to model the probability](https://arxiv.org/abs/1805.12295)[1](https://arxiv.org/abs/1805.12295). This model is simple and fast, but ignores the spatial and channel-wise correlations in the latent representation.
- [The **hyper prior** model introduces a second latent representation (called hyper features) that encodes the statistics of the first latent representation, and uses it to condition the entropy model](https://arxiv.org/abs/2202.05492)[2](https://arxiv.org/abs/2202.05492). This model can capture some spatial correlations, but still assumes that each channel of the latent representation is independent.
- [The **auto-regressive prior** model uses a convolutional neural network to predict the probability of each element of the latent representation based on its causal context (i.e., previously decoded elements) in both spatial and channel dimensions](https://arxiv.org/abs/2211.07273)[3](https://arxiv.org/abs/2211.07273). This model can capture both spatial and channel-wise correlations, but is slower and more complex than the previous models.
- [The **mixture Gaussian model** uses a mixture of Gaussian distributions to model the probability of each element of the latent representation, and uses an attention module to select the most suitable mixture component based on the local context](https://paperswithcode.com/paper/channel-wise-autoregressive-entropy-models)[4](https://paperswithcode.com/paper/channel-wise-autoregressive-entropy-models). This model can adapt to different image regions and capture local dependencies, but also increases the complexity and computation cost.



**how does the entropy model estimate the bits rate in learned image compression**

The entropy model estimates the bits rate in learned image compression by predicting the probability distribution of the quantized latent representation. The bits rate is calculated by the **cross entropy** between the predicted distribution and the actual distribution of the latent representation. The cross entropy can be approximated by the **negative log-likelihood** of the latent representation given the predicted distribution. Mathematically, the bits rate can be expressed as:

$$R = -\mathbb{E}_{\mathbf{y} \sim p(\mathbf{y}|\mathbf{x})}[\log_2 p(\mathbf{y}|\mathbf{x})]$$

where $\mathbf{x}$ is the input image, $\mathbf{y}$ is the quantized latent representation, and $p(\mathbf{y}|\mathbf{x})$ is the predicted distribution by the entropy model¹. The entropy model can be implemented by various neural network architectures, such as convolutional neural networks², hyperprior models³, or transformer-based models. The entropy model can be trained jointly with the encoder and decoder networks by minimizing the **rate-distortion loss**, which balances the bits rate and the reconstruction error.

Source: Conversation with Bing, 2023/5/16
(1) MLIC: Multi-Reference Entropy Model for Learned Image Compression. https://arxiv.org/pdf/2211.07273.
(2) MLIC: Multi-Reference Entropy Model for Learned Image Compression. https://arxiv.org/abs/2211.07273.
(3) Entroformer: A Transformer-based Entropy Model for Learned Image .... https://arxiv.org/abs/2202.05492.









