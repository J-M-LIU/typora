## 背景

随着数字媒体技术的飞速发展，视频内容已成为互联网上最重要的数据形式之一。高效的视频编码对于带宽利用和数据存储具有重大意义，尤其在在线视频流、视频会议和数字电视等应用中尤为突出。

在过去几十年中，已经发展出多代视频编码标准，比如H.264/AVC、H.265/HEVC、H.266/VVC，这些传统编码标准都是基于预测编码范式，依靠手动设计的模块，例如基于块的运动估计和离散余弦变换（DCT）来减少视频序列中的冗余。尽管各模块都经过精心设计，但整个压缩系统并未进行端到端优化。

近年来，neural video codec已经探索了一个全新的发展方向。现有的深度视频压缩工作可分为无时延约束和时延约束两类。对于无时延约束编码，参考帧可以来自于当前帧之后的位置。由于存在多重参考帧，这种编码方式存在较大的延迟，且显著增加内存开销和计算消耗。对于低时延约束下的编码，参考帧来自于之前的重构帧。

————————————————————————————————————————————

在设计视频编码模型以应对低延迟应用场景时，我们面临一个关键挑战：这些模型常常需要在资源受限的移动设备上部署。为了适应这些设备的硬件限制，模型的压缩和加速变得至关重要，以确保能够高效地执行推理任务。然而，现有的视频编码模型往往对计算资源的需求极高，这在移动和低功耗设备上尤其成问题。








