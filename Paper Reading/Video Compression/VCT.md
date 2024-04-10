# VCT: A Video Compression Transformer



## INTRO

已有的大部分神经网络视频编码方案与传统编码方法类似，均依赖于运动预测，然后通过变形操作将预测流与之前的重建结合，并计算残差。多种复杂模块之间的连接非常繁琐，限制网络只能在匹配架构偏差的数据上工作，并没有解决“hand-crafted”的问题。

本文提出基于transformer的时间熵模型取代光流预测、扭曲和残差补偿。由此产生的视频压缩transformer (VCT)在标准视频压缩数据集上优于之前的方法，同时不受其架构偏差和先验的影响。创建了合成数据来探索架构偏差的影响，并表明在架构组件设计的类型视频上(静态帧上的平移或模糊)，与以前的方法相比是有利的，尽管transformer不依赖于这些组件中的任何一个。更重要的是，在没有明显匹配的架构组件(场景之间的锐化、渐隐)的视频上，优于之前的方法，显示了删除手工制作元素的好处，并让transformer从数据中学习一切。

transformer压缩视频分为两个步骤：首先，使用有损变换编码[3]将帧从图像空间映射到量化表示，独立地为每一帧；之后，利用transformer对表示的分布进行建模，以无损的方式压缩量化后的表示。transformer预测的分布越好，所需存储的比特数就越少。

这种设置避免了复杂的状态转换或变形操作，让transformer学会利用帧之间的任意关系。并且，摆脱了时间误差传播，因为当前帧重建不依赖于之前的重建帧。相比之下，warp-based的方法中，$\hat{x_i}$ 是 

与基于扭曲的方法相比，xiˆ是扭曲xi−1ˆ的函数，这意味着xiˆ中的任何视觉错误都将向前传播，并需要额外的比特来用残差纠正。
