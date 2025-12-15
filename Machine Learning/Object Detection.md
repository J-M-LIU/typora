# 目标检测



## 特征金字塔

<img src="https://pic3.zhimg.com/v2-65096e0b4b76372b2926bd412b749b6e_1440w.jpg" style="zoom:67%;" />

### 通过图像金字塔构建特征金字塔

<img src="https://pica.zhimg.com/v2-fdcd9e60c572b74d3a177156d3a85b8e_1440w.jpg"  />

对每一种尺度的图像进行特征提取，能够产生多尺度的特征表示，并且所有等级的特征图都具有较强的语义信息，甚至包括一些高分辨率的特征图。

缺点：

1. 推理时间大幅度增加；

2. 由于内存占用巨大，用图像金字塔的形式训练一个端到端的深度神经网络变得不可行；



### Faster-R-CNN中的特征图预测

![](https://pic2.zhimg.com/v2-924d8c3f8961b4d2e9ca3191af5c82a5_1440w.jpg)

利用单个高层特征图进行预测。

例如Faster R-CNN中的RPN层就是利用单个高层特征图进行物体的分类和bounding box的回归。



###  金字塔型特征层级 ConvNet's pyramidal feature hierarchy

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20241208211559321.png" alt="image-20241208211559321" style="zoom:67%;" />

SSD one-stage目标检测模型：避免使用低层特征图，放弃了重用已经计算的层，而是从网络的高层开始构建金字塔（例如，VGG网络的Conv4之后，再添加几个新的卷积层），因此，SSD错过了重用低层高分辨的特征图，即没有充分利用到低层特征图中的空间信息(这些信息对小物体的检测十分重要)。

### 特征金字塔 Feature Pyramid Networks

<img src="https://pic1.zhimg.com/v2-d136addbe3dc08f6cbae7233f753bf9a_1440w.jpg" style="zoom:67%;" />