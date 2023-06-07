# Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression



## 引言

​	近年来，基于深度学习的图像编解码技术蓬勃发展。已有的研究工作多集中于熵模型的设计上，以准确预测量化隐表示的概率分布，如因子分解模型[^1]、超先验模型[^2]、自回归先验[^3]、混合高斯模型[^4]、基于transformer的模型[^5]等。受神经图像编解码器成功的启发，近年来神经视频编解码器受到越来越多的关注。

​	现有的神经视频编解码器工作大致可以分为三类：基于残差编码、基于条件编码和基于3D autoencoder 的解决方案。较多方法基于残差编码，来自传统的视频编解码器。即首先运动补偿生成预测帧，然后将预测帧与当前帧的残差进行编码。基于条件编码的解决方案[^6-9]，将时间帧或特征作为当前帧编码的条件。与残差编码相比，条件编码具有更低的信息熵[^6]。基于3D autoencoder的解决方案是对神经图像编解码器的自然延伸，通过扩大输入维数。但它带来了较大的编码延迟，并显著增加了内存成本。

​	因此，本文提出了一种综合熵模型，可以有效地利用空间和时间关系。介绍了潜先验和对偶空间先验。潜在先验探索了跨帧的隐表示的时间相关性。

## 相关工作

​	熵模型的设计成为研究的热点。Ballé等[^1]提出了分解模型，得到了比jpeg2000更好的R-D性能。超先验[8]引入了分层设计，并使用额外的比特来估计分布，得到了与H.265相当的结果。随后，提出了自回归先验[36]来探索空间相关性。利用超先验，获得了较高的压缩比。然而，自回归先验推理在一个序列化的顺序，所有的空间位置，因此它是相当慢。棋盘上下文模型[16]通过引入两步编码解决了复杂性问题。此外，为了进一步提高压缩比，提出了混合高斯模型[12]，其RD结果与H.266相当。近年来，视觉转换器备受关注，其对应的熵模型[23]帮助神经图像编解码器超越H.266内编码。



## 参考文献

[^1]: Johannes Ballé, Valero Laparra, and Eero P. Simoncelli. 2017. **End-to-end Optimized Image Compression.** 
[^2]: Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. 2018. **Variational image compression with a scale hyperprior.**
[^3]: David Minnen, Johannes Ballé, and George D Toderici. 2018. **Joint autoregressive and hierarchical priors for learned image compression.**
[^4]: Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro Katto. 2020. **Learned image compression with discretized gaussian mixture likelihoods and attention modules.**
[^5]: A Burakhan Koyuncu, Han Gao, and Eckehard Steinbach. 2022. **Contextformer: A Transformer with Spatio-Channel Attention for Context Modeling in Learned Image Compression.**
[^6]: Théo Ladune, Pierrick Philippe, Wassim Hamidouche, Lu Zhang, and Olivier Déforges. 2020. **Optical Flow and Mode Selection for Learning-based Video Coding.**
[^7]: Théo Ladune, Pierrick Philippe, Wassim Hamidouche, Lu Zhang, and Olivier Déforges. 2021. Conditional Coding and Variable Bitrate for Practical Learned Video Coding.
[^8]: Théo Ladune, Pierrick Philippe, Wassim Hamidouche, Lu Zhang, and Olivier Déforges. 2021. **Conditional Coding for Flexible Learned Video Compression.**    
[^9]: JiahaoLi, BinLi, and Yan Lu.2021. **Deep contextual video compression.**

