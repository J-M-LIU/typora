**A.Overview**
Aiming to find a better way to learn and utilize temporal contexts, we propose a new learned video compression scheme based on temporal context mining and re-filling. An overview of our scheme is depicted in Fig.[2]

1. Motion Estimation: We feed the current frame $x_t$ and the previously decoded frame $\hat{x}_{t-1}$ into a neural network-based motion estimation module to estimate the optical flow. The optical flow is considered as the estimated motion vector (MV) $v_t$ for each pixel. In this paper, the motion estimation module is based on the pre-trained Spynet [44].

2) $MVEncoder$-Decoder: After obtaining the motion vec- tors $v_t,$ we use the MV encoder and decoder to compress and reconstruct the input MV $v_t$ in a lossy way. Specifically, $v_t$ is compressed by an auto-encoder with the hyper prion structure [45]. The reconstructed MV is denoted as $\hat{v}_t.$

3) Learned Temporal Contexts: We propose a TCM module to learn richer and more accurate temporal contexts from the propagated feature $F_{t-1}$ instead of the previously decoded frame $\hat{x}_{t-1}.$ Instead of producing only a single-scale temporal context, the TCM module generates multi-scale temporal con- texts $\bar{C}_t^l$ to capture spatial-temporal non-uniform motion and texture, where $l$ is the index of different scales. The learned temporal contexts are re-filled in the contextual encoder- decoder, the frame generator, and the temporal context encoder to help improve the compression ratio. This procedure is referred to as TCR. We will introduce them in detail in Section III-B and III-CI

4) Contextual Encoder-Decoder and Frame Generator:
   With the assistance of the re-filled multi-scale temporal con- texts $\bar{C}_t^t,$ the contextual encoder-decoder and the frame gen- erator are used to compress and reconstruct the current frame $x_t.$ We denote the decoded frame as $\hat{x}_t$ and the feature before obtaining $\hat{x}_t\mathrm{~as~}F_t.\hat{x}_t\mathrm{~and~}F_t$ are propagated to help compress the next frame $x_{t+1}.$ The details are presented in Section III-C

5) Temporal Context Encoder: To utilize the temporal correlation of the latent representations of different frames produced by the contextual encoder, we use a temporal context encoder to generate the temporal prior by taking advantage of the multi-scale temporal contexts $\bar{C}_t^l$.More information is provided in Section III-C

6) Entropy Model: We use the factorized entropy model for hyper prior and Laplace distribution to model the latent representations as [27]. We do not apply the auto-regressive entropy model to make the decoding processes parallelization- friendly, even though it is helpful to improve the compression ratio. For the contextual encoder-decoder, we fuse the hyper and temporal priors [27] to estimate more accurate parameters of Laplace distribution. When writing bitstream, we apply a similar implementation as in [46].

B. Temporal Context Mining

Considering that the previously decoded frame $\hat{x}_{t-1}$ loses much information as it only contains three channels, it is not optimal to learn temporal contexts from $\hat{x}_{t-1}.$ In our paper, we propose a TCM module to learn temporal contexts from the propagated feature $F_{t-1}.$ Different from the existing video compression schemes in feature domain $[10],[11],[47]$, which extract features from the previously decoded frame $\hat{x}_{t-1},$ we propagate the feature $F_{t-1}$ before obtaining $\hat{x}_{t-1}$. Specifically, in the reconstruction procedure of $\hat{x}_{t-1},$ we store the feature $F_{t-1}$ before the last convolutional layer of the frame generaton in the generalized decoded picture buffer (DPB). To reduce the computational complexity, instead of storing multiple features of previously decoded frames [10], [11], we only store a single feature. Then $F_{t-1}$ is propagated to learn temporal contexts for compressing the current frame $\hat{x}_t.$ For the first P frame, we still extract the feature from the reconstructed I frame, to make the model adapt to different image codecs.

Besides, learning a single scale context may not describe the spatio-temporal non-uniform motion and texture well [30]- [32]. As shown in Fig.[3] in the largest-scale context, some channels focus on the texture information and some focus on the color information. In the smallest-scale context, channels mainly focus on the regions with large motion. Therefore, a hierarchical approach is performed to learn multi-scale temporal contexts.

As shown in Fig. $4$ we first generate multi-scale features $F_{t-1}^l$ from the propagated feature $F_{t-1}$ using a feature ex- traction module $(extract)$ with $L$ levels which consists of convolutional layers and residual blocks [48] (three levels are used in our paper).

$$
F_{t-1}^l=extract\left(F_{t-1}\right),l=0,1,2
$$
 Meanwhile, the decoded MV $\hat{v}_t$ are downsampled using the bilinear filter to generate multi-scale MVs $\hat{v}_t^l,$ where $\hat{v}_t^0$ is set to $\hat{v}_t.$ Note that each downsampled MV is divided by 2. Then, we warp $(warp)$ the multi-scale features $F_{t-1}^l$ using the associated MV $\hat{v}_t^l$ at the same scale.

$$
\bar{F}_{t-1}^l=warp\left(F_{t-1}^l,\hat{v}_t^l\right),l=0,1,2
$$
 After that, we use an upsample $(upsample)$ module, consisting of one subpixel layer [49] and one residual block, to upsample $\bar{F}_{t-1}^{l+1}.$ The upsampled feature is then concatenated $(concat)$ with $\bar{F}_{t-1}^l$ at the same scale.

$$
\tilde{F}_{t-1}^l=concat\left(\bar{F}_{t-1}^l,upsample\left(\bar{F}_{t-1}^{l+1}\right)\right),l=0,1
$$
 At each level of the hierarchical structure, a context refine-
 ment module $(refine)$, consisting of one convolutional layer and one residual block, is used to learn the residue [50]. The residue is added to $\bar{F}_{t-1}^l$ to generate the final temporal contexts $\bar{C}_t^l,$ as illustrated in Fig $\color{red}{\boxed{4}}$
$$
\bar{C}_t^l=\bar{F}_{t-1}^l+refine\left(\tilde{F}_{t-1}^l\right),l=0,1,2
$$
$C.$ Temporal Context Re-filling
To fully take advantage of the temporal correlation, we re- fill the learned multi-scale temporal contexts into the modules of our compression scheme, including the contextual encoder- decoder, the frame generator, and the temporal context en- coder, as shown in Fig.[5] The temporal context plays an important role in temporal prediction and temporal entropy modeling. With the re-filled multi-scale temporal contexts, the compression ratio of our scheme is improved a lot.

1. Contextual Encoder-Decoder and Frame Generator:
   We concatenate the largest-scale temporal context $\bar{C}_t^0$ with the current frame $x_t$, and then feed them into the contextual encoder. In the process of mapping from $x_t$ to the latent representation $f_y$, we also concatenate $\bar{C}_t^1$ and $\bar{C}_t^2$ with other scales into the encoder. Symmetric with the encoder, the contextual decoder maps the quantized latent representation $\hat{f}_y$ to the feature $\hat{F}_t$ with the assistance of $\bar{C}_t^1$ and $\bar{C}_t^2$. Then $\hat{F}_t$ and $\bar{C}_t^0$ are concatenated and fed into the frame generaton to obtain the reconstructed frame $\hat{x}_t.$ Considering that the concatenation increases the number of channels, a “bottleneck” building residual block is used to reduce the complexity of the middle layer. The detailed architectures are illustrated in Fig.[6] The feature $F_t$ before the last convolutional layer of the frame generator is propagated to help compress the next frame $x_{t+1}.$

2) Temporal Context Encoder: To explore the temporal correlation of the latent representations of different frames, we use a temporal context encoder to obtain the lower- dimensional temporal prior $f_c.$ Instead of using a single- scale temporal context as $[27]$, we concatenate the multi-scale temporal contexts in the process of generating temporal prior. The temporal prior is fused with the hyper prior to estimate the means and variance of Laplacian distribution for the latent representation $\hat{f}_y.$ The architecture of the temporal context encoder is presented in Fig. [5]

 $D.\:Loss\:Function$

Our scheme targets to jointly optimize the rate-distortion
(R-D) cost.

$$
L_t=\lambda D_t+R_t=\lambda d(x_t,\hat{x}_t)+R_t^{\hat{v}}+R_t^{\hat{f}}
$$
$L_t$ is the loss function for the current time step $t.d(x_t,\hat{x}_t)$ refers to the dístortion between the input frame $x_t$ and the reconstructed frame $\hat{x}_t$, where $d(\cdot)$ denotes the mean-square- error or 1-MS-SSM [39]. $R_t^{\hat{v}}$ represents the bit rate used fon encoding the quantized motion vector latent representation and the associated hyper prior. $R_t^{\hat{f}}$ represents the bit rate used fon encoding the quantized contextual latent representation and the associated hyper prior. We train our model step-by-step to make the training more stable as previous the scheme [9]. It is worth mentioning that, in the last five epochs, we use a commonly-used training strategy in recent papers [8], [16]. $[43],[51],$ that trains our model on the sequential training frames to alleviate the error propagation.

$$
L^{T}=\frac{1}{T}\sum_{t}L_{t}=\frac{1}{T}\sum_{t}\left\{\lambda d\left(x_{t},\hat{x}_{t}\right)+R_{t}^{\hat{v}}+R_{t}^{\hat{f}}\right\}
$$
 where $T$ is the time interval and set as 4 in our experiments.

