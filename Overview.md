在深度神经网络中，常用的卷积块包括以下几种：

基本卷积块（BASIC Block）：由两个3x3的卷积层和一个ReLU激活函数组成，是最简单的卷积块之一。
瓶颈卷积块（BOTTLENECK Block）：由一个1x1的卷积层、一个3x3的卷积层和一个ReLU激活函数组成，可用于减少网络参数和提高模型性能。
Inception模块：由多个并行的卷积层组成，每个并行卷积层有不同的卷积核大小和数量，能够提取多种不同大小的特征。
残差卷积块（Residual Block）：由两个或三个卷积层、Batch Normalization和跨层连接组成，能够有效地解决深度神经网络中的梯度消失问题。
向量化卷积块（Vectorized Convolution Block）：是一种多分支卷积块，可以在不同的分支中对不同的特征进行处理。
深度可分离卷积块（Depthwise Separable Convolution Block）：由深度卷积和逐点卷积两部分组成，可在减少参数数量的同时，提高模型性能。
除了以上常用的卷积块之外，还有一些变种和扩展，如ResNeXt、SENet、DenseNet等。这些卷积块的设计都旨在提高深度神经网络的表达能力和性能。


BASIC和BOTTLENECK是深度神经网络中常用的两种基本的卷积块（Convolutional Block）。

BASIC块通常由两个卷积层组成，其中第一个卷积层的卷积核大小为3x3，第二个卷积层的卷积核大小也为3x3。该卷积块的主要目的是对输入数据进行特征提取，增加网络深度，提高网络的表达能力。

BOTTLENECK块则更为复杂，通常由三个卷积层组成，其中第一个卷积层的卷积核大小为1x1，第二个卷积层的卷积核大小为3x3，第三个卷积层的卷积核大小也为1x1。该卷积块可以在减少参数数量的同时，增加网络的深度和表达能力。

在一些深度神经网络中，BASIC和BOTTLENECK块被广泛使用，如ResNet、DenseNet等。 作者：熊二爱光头强丫 https://www.bilibili.com/read/cv22705702/ 出处：bilibili
