## Introduction

已有的SR方法无法直接应用于移动设备等，因为模型参数量很大，且直接量化到uint8后性能下降严重。除此之外也存在硬件的设计限制。

**解决了什么问题：**移动端设备上的实时图像超分辨重建，并解决模型的全量化部署问题。

**Trick：**使用 **clipped relu**

## Related Work

**紧凑网络设计**

用于对象检测的**SqueezeNet**，在ImageNet上实现AlexNet性能，参数数量减少了50倍；**MobileNetV1**使用深度可分离卷积，获得比SqueezeNet更好的性能；**ShuffleNet**通过使用**分组卷积和信道变换**算子进一步提高了ImageNet性能，更轻量；与ShuffleNet类似，DeepRoots提出使用1x1卷积来代替信道变换。对于人脸验证，MobileFaceNets使用深度可分离卷积和瓶颈层来构建特定于应用程序的轻量级网络。对于图像超分辨率，**IMDN**使用信道分割来构建一个更轻的网络。与以前的方法不同，**NASNet通过最优结构搜索，超越了许多最先进的方法。





