# ELF-VC: Efﬁcient Learned Flexible-Rate Video Coding



## INTRO

本文提出一个灵活的bit-rate框架，允许单个模型覆盖大而密集的比特率范围，而计算和参数数量的增加可以忽略不计；向ml编解码器优化的高效骨干网络以及一种新的在线流量预测方案，该方案利用先验信息以更有效地压缩，可以提高低延迟模式(仅I帧和Pframes)的性能，同时大大提高计算效率。

在流行的视频测试集 UVG 和 MCL-JCV 上，以PSNR、MS-SSIM和VMAF为指标对所提出方法进行了基准测试；在UVG上，与H.264相比BD-rate降低了44%，与H.265相比降低了26%，与AV1相比降低了15%，与目前最好的ML编码器相比降低了35%。同时，在NVIDIA Titan V GPU上，该方法编码/解码VGA的速度为49/91 FPS, HD 720的速度为19/35 FPS, HD 1080的速度为10/18 FPS。