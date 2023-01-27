# 基于深度学习的视频压缩

## 引言

​	视频数据具有连续性强、数据量大的特点，因此视频压缩技术对于视频的储存与传输具有重要意义。根据压缩中利用的参考信息不同，视频压缩可以分成帧内压缩与帧间压缩。帧内压缩只利用当前帧的帧内信息进行压缩，可以视为图像压缩。帧间压缩利用已经解码的参考帧信息，消除时间冗余，达到压缩的目的。
​	视频压缩研究中关注的一个重点是率-失真的权衡(rate-dissortion tradeoff)，即在一定的码率限制下，减少视频的失真；在允许一定的失真下，获得更低的码率。传统视频编码采用基于块划分的混合编码框架，包括帧内预测、帧间预测、变换、量化、熵编码和环路滤波等技术模块。这些模块经过几十年的发展已逐渐成熟。
​	 基于深度学习的视频编码的主要研究方向目前为：
​	（1）结合深度神经网络的传统视频编码，即对传统视频编码框架中的模块引入深度学习的优化方法；
​	（2）完全基于深度学习的压缩框架。



## 完全基于深度学习的视频压缩

### 端到端的视频编码方案

​	由于传统视频压缩框架的每一个模块之间密切联系，基于深度学习的算法对某个模块的优化与改进，尽管可以
提升该模块的压缩性能，但是对整体性能的影响则相对较小。而端到端的视频编码中，所有模块都是基于深度神经网络实现，可以直接端到端优化率失真目标函数，更容易实现全局最优。

#### **Lu, Guo, et al. "DVC: An end-to-end deep video compression framework." in CVPR. 2019.**

<img src="https://img-blog.csdnimg.cn/20201010163822194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x3cGx3Zg==,size_16,color_FFFFFF,t_70#pic_center" style="zoom:55%;" />

​	Lu提出了首个端到端的基于学习的视频编码框架DVC，其与传统视频编码框架中的模块存在一一对应的关系。首先使用光流网络估计当前帧与参考帧的运动信息，随后对运动信息进行编码。然后将参考帧、补偿后的参考帧以及压缩后的光流输入运动补偿网络，得到预测帧。最后再对预测帧与当前帧的残差进行编码。整个网络使用率失真函数进行优化，码流包括压缩光流的比特与压缩残差的比特两部分。根据文献里的描述，DVC的性能超越了H.264，在MS-­SSIM作为失真衡量指标时，压缩性能优于H.265。





▪ 基于深度学习的压缩视频质量增强研究 DNN-based enhancement for compressed video

▪ 结合深度网络的传统视频压缩方法 DNN-based handcrafted video compression

▪ 端到端优化的视频压缩深度网络
 End-to-end deep video compression network

▪ 兼容一般播放器的深度学习压缩方法
 Learning for compression with standard decoder



## 端到端的视频压缩

### 面向随机切入场景

1. Djelouah A，Campos J，Schaub-Meyer S and Schroers C. 2019. Neural inter-frame compression for video coding. 首先使用光流网络提取前后向参考帧相对于当前帧的原始运动场，然后将原始运动场与对齐 的参考帧( warped frame) 和原始帧一起输入自编码 器进行压缩，解码得到的运动场和融合系数( blend- ing coefficients)通过内插方式得到预测帧。然后在隐空间( latent space) 进行残差的提取和编码。该方 案在峰值信噪比( peak-signal-noise-ratiom，PSNR) 上的压缩性能与H. 265相当。
2. Park W and Kim M. 2019. Deep predictive video compression with bi-directionalprediction. 使用基于帧内插的双向编码结构，并且在编码端和解 码端同时引入运动估计网络，从而不再传输运动信息，在低分辨率视频上的压缩性能与 H.265 相当。
3. Yang R，Mentzer F，van Gool L and Timofte R. 2020a. Learning for video compression with hierarchical quality and recurrent enhancement. 提出包含 3 个质量层的分层双向预测视频压缩方案，结合使用回归的质量增强网络，取得了与H.265相近的压缩性能。
4. Yilmaz M A and Tekalp A M. 2020. End-to-end rate-distortion optimiza- tion for bi-directional learned video compression. 提出一个端到端优化的分层双向预测视频 压缩方案，在 PSNR 上的压缩性能接近 H. 265。
5. Pessoa J，Aidos H，Toms P and Figueiredo MAT. 2020. End-to-end learning of video compression using spatio-temporal autoencoders. 将视频压缩问 题看成一个“空—时”域自编码器的率失真优化问题，从而避免了显式的运动估计和补偿。



### 面向低延时场景

1. Rippel O，Nair S，Lew C，Branson S，Anderson A and Bourdev L. Learnedvideocompression. 首个面向低延时的端到端视频压缩方案。在解码器端使用隐状态( latent state) 保存历史帧的信息，并使用生成网络生成运动场和残差的重建，设计了空域码率控制方法减小误差累积，在MS-SSIM上性能与H.265相当。
2. Agustsson E，Minnen D，Johnston N，Ballé J，Hwang S J and Toderici G. 2020. Scale-space flow for end-to-end optimized video compression. 使用自编码器 基于前一个参考帧和当前原始帧提取原始的尺度空 间流(scale-spaceflow)，并进行压缩编码，解码得到 的尺度空间流结合参考帧的高斯金字塔通过三线性 插值( trilinear interpolation) 操作得到当前帧的预测，剩下的残差使用另外一个自编码器进行压缩，该方案的压缩性能与H.265相当。
3. Yang R，Mentzer F，van Gool L and Timofte R. 2020b. Learning for video compression with recurrent auto-encoder and recurrent proba- bility model. 
4. Golinski A，Pourreza R，Yang Y，Sautiere G and Cohen T S. 2020. Feedback recurrent autoencoder for video compression. 提出一个基于自编码器的端到端视频压缩方案，在解码端使用反馈回归模块将提取的历史隐 变量信息反馈回编码端，且显式地使用运动估计模块辅助运动信息的提取和压缩，该方案在MS-SSIM 上的压缩性能超过了 H.265。
5. Lu G，Ouyang W L，Xu D，Zhang X Y，Cai C L and Gao Z Y. 2019. DVC: an end-to-end deep video compression framework. 第一个端到端视频压缩方案。该方案可以看成是传统视频压缩方案的深度学习版 本，在 PSNR 和 MS-SSIM 上的压缩性能分别与 H.264 和 H.265相当。由于灵活性和高性能，该方 案成为本领域基线方案，常在后续工作中引用。
6. Liu H J，Shen H，Huang L C，Lu M，Chen T and Ma Z. 2020a. Learned video compression via joint spatial-temporal correlation exploration. 结合时域先验、空域自回归先验和空域超先验来压缩运动信息，在 MS-SSIM 指标压缩性能上超过了DVC 的方法。

### 预测编码框架和残差压缩

1. G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, and Z. Gao, “DVC: an end-to-end deep video compression framework,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11006–11015, 2019.
2. G. Lu, C. Cai, X. Zhang, L. Chen, W. Ouyang, D. Xu, and Z. Gao, “Content adaptive and error propagation aware deep video compression,” in European Conference on Computer Vision, pp. 456–472, Springer, 2020.

2. R. Yang, F. Mentzer, L. V. Gool, and R. Timofte, “Learning for video compression with hierarchical quality and recurrent enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

3. J. Lin, D. Liu, H. Li, and F. Wu, “M-LVC: multiple frames prediction for learned video compression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

4. A. Djelouah, J. Campos, S. Schaub-Meyer, and C. Schroers, “Neural inter-frame compression for video coding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.
5. E. Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, and G. Toderici, “Scale-space ﬂow for end-to-end optimized video compression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8503–8512, 2020.
6. C.-Y. Wu, N. Singhal, and P. Krähenbühl, “Video compression through image interpolation,” in ECCV, 2018.
7. R. Yang, Y. Yang, J. Marino, and S. Mandt, “Hierarchical autoregressive modeling for neural video compression,” 9th International Conference on Learning Representations, ICLR, 2021.

### 条件编码 conditional coding

1. J. Liu, S. Wang, W.-C. Ma, M. Shah, R. Hu, P. Dhawan, and R. Urtasun, “Conditional entropy coding for efﬁcient video compression,” arXiv preprint arXiv:2008.09180, 2020.
2. O. Rippel, S. Nair, C. Lew, S. Branson, A. G. Anderson, and L. Bourdev, “Learned video compression,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3454–3463, 2019.
3. T. Ladune, P. Philippe, W. Hamidouche, L. Zhang, and O. Déforges, “Conditional coding for ﬂexible learned video compression,” in Neural Compression: From Information Theory to Applications – Workshop @ ICLR, 2021.
4. T. Ladune, P. Philippe, W. Hamidouche, L. Zhang, and O. Déforges, “Optical ﬂow and mode selection for learning-based video coding,” in 22nd IEEE International Workshop on Multimedia Signal Processing, 2020.



## 可参考论文总结

1. **DVC**:  GuoLu, WanliOuyang, DongXu, XiaoyunZhang, ChunleiCai,and ZhiyongGao. 2019. *DVC: an end-to-end deep video compression framework.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11006– 11015.

2. **DVCPro**: GuoLu, XiaoyunZhang, WanliOuyang, LiChen, ZhiyongGao,and DongXu. 2020. *An end-to-end learning framework for video compression.* IEEE transactions on pattern analysis and machine intelligence (2020).

3. **RY_CVPR20**: R. Yang, F. Mentzer, L. V. Gool, and R. Timofte. *Learning for video compression with hierarchical quality and recurrent enhancement*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

4. **LU_ECCV20**: G. Lu, C. Cai, X. Zhang, L. Chen, W. Ouyang, D. Xu, and Z. Gao, “*Content adaptive and error propagation aware deep video compression*,” in European Conference on Computer Vision, pp. 456–472, Springer, 2020.

5. **HU_ECCV20**: Z. Hu, Z. Chen, D. Xu, G. Lu, W. Ouyang, and S. Gu, “*Improving deep video compression by resolutionadaptive ﬂow coding*,” in European Conference on Computer Vision, pp. 193–209, Springer, 2020.

6. **Liu_AAAI_20: **Liu *et al.*, 2020 Haojie Liu, Lichao Huang, Ming Lu, Tong Chen, and Zhan Ma. *Learned video compression via joint spatial-temporal correlation exploration.* In *AAAI*, 2020.

7. **SSF：**Eirikur Agustsson et al. “*Scale-space flow for end-to-end optimized video compression*”. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020, pp. 8503–8512.

8. **M-LVC**: Jianping Lin, Dong Liu, Houqiang Li, and Feng Wu. 2020. *M-LVC: multiple frames prediction for learned video compression.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

9. **PLVC**: [Liu *et al.*, 2020] Haojie Liu, Lichao Huang, Ming Lu, Tong Chen, and Zhan Ma. *Learned video compression via joint spatial-temporal correlation exploration.* In *AAAI*, 2020.

10. **RLVC**: Ren Yang, Fabian Mentzer, Luc Van Gool, and Radu Timofte. 2020. *Learning for Video Compression with Recurrent Auto-Encoder and Recurrent Probability Model.*

11. **DCVC:** Jiahao Li, Bin Li, and Yan Lu. “*Deep contextual video compression*”. In: *Advances in Neural*

    *Information Processing Systems* 34 (2021). 

12. **Sheng 2021**: Xihua Sheng, Jiahao Li, Bin Li, Li Li, Dong Liu, and Yan Lu. 2021. *Temporal Context Mining for Learned Video Compression.*

13. **ELF-vc:** Oren Rippel et al. “Elf-vc: Efficient learned flexible-rate video coding”. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021, pp. 14479–14488.

14. **FVC: **Zhihao Hu, Guo Lu, and Dong Xu. “FVC: A new framework towards deep video compression

    in feature space”. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021, pp. 1502–1511.

15. **VCT**: A Video Compression Transformer.

16. Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression.ACM MM 2022









tips：

​	这个学期主要是课程比较多，想着把时间主要留给下学期，所以上一个多月主要就是在完成考试和大作业的事情；

​	然后起初是考虑到有些基础的东西还不太会，就重点把花书和西瓜书这一类的内容过了一遍，然后看了看信息论 视频处理相关的内容；之后主要是看了些论文，然后写了点读书报告，这段时间重点主要是放在端到端的视频编码上，目前没有太关注编码中某一模块的具体优化和改进；但目前感觉还是对整体的一个具体研究的切入点不是很明确，所以想再多看些文章后总结个综述出来。



​	<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230115152714112.png" alt="image-20230115152714112" style="zoom:30%;" />

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230115152947947.png" alt="image-20230115152947947" style="zoom:30%;" />

​	已有的一些工作主要差别在其采用的编码框架上，一种就是预测编码：参考帧仅来自于前一帧，对生成的预测帧和当前帧间的残差进行编码；另一种则是信息熵小于残差编码信息熵的条件编码，从微软提出的DCVC开始，引入基于条件编码的自编码器。还有已有的工作是对多帧进行编码，但是时延较高，不适于实时场景。

现有的端到端的编解码器大多专注于如何生成优化的潜在编码，然后设计其中的网络结构。并关注如何有效利用空间和时间相关性进行编码。以往的基于残差编码的结构没有结合空间和时间上的先验性，

在本文中，我们关注的是通过有效地利用空间和时间相关性来设计熵模型。事实上，一些著作也已经开始对此进行研究。例如，[31]中提出了条件熵编码。在[27,41]中使用了从时态特征中提取的时态上下文。Yang et al.[49]提出了循环熵模型。然而，这些工作[27,31,41,49]更多地关注于利用时间相关性。虽然[27]中的工作也研究了空间相关性，但使用了自回归先验，导致编码速度非常慢。我们需要一个既能充分利用时空相关性又具有较低复杂度的熵模型。

