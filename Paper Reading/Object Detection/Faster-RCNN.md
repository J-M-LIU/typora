# Faster-RCNN

​	Faster-RCNN将feature提取、proposal提取、bounding box regressinon(rect refine)和classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。

<img src="https://img-blog.csdn.net/20180424111008306?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pOaW5nV2Vp/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:67%;" />

## Two-Stage/One-Stage

目标检测的主流方式：

- 生成一系列的候选框，这些候选框称为proposals；
- 根据候选框判断候选框里的内容是前景还是背景，即是否包含了检测物体；
- 用回归的方式去微调候选框，使其更准确地框取物体，这过程我们称bounding box regression；

### 两阶段

1. 先提取物体区域，生成候选框(region proposals)，即通过第一阶段的网络回归出目标框的大概位置、大小及是前景的概率；
2. 对这些候选框进行分类，计算物体类别。

### 一阶段

​	直接通过网络回归出物体位置和类别。

### 一阶段和两阶段区别

两阶段目标检测算法比单阶段算法相对来说精度更加高，目标检测两阶段比一阶段的算法精度高的原因有以下几点：

**1. 正负样本的不均衡性**

当某一类别的样本数特别多的时候，训练出来的网络对该类的检测精度往往会比较高。而当某一类的训练样本数较少的时候，模型对该类目标的检测精度就会有所下降，这就是所谓样本的不均衡性导致的检测精度的差异。

对于一阶段的目标检测来说，它既要做定位又要做分类，最后几层中1×1的卷积层的loss都混合在一起，没有明确的分工哪部分专门做分类，哪部分专门做预测框的回归，这样的话对于每个参数来说，学习的难度就增加了。

对于二阶段的目标检测来说(Faster RCNN)，在RPN网络结构中进行了前景和背景的分类和检测，这个过程与一阶段的目标检测直接一上来就进行分类和检测要简单的很多，有了前景和背景的区分，就可以选择性的挑选样本，是的正负样本变得更加的均衡，然后重点对一些参数进行分类训练。训练的分类难度会比一阶段目标检测直接做混合分类和预测框回归要来的简单很多。

**2. 样本的不一致性**

怎么理解样本不一致性呢？首先我们都知道在RPN获得多个anchors的时候，会使用一个NMS。在进行回归操作的时候，预测框和gt的IoU同回归后预测框和gt的IOU相比，一般会有较大的变化，但是NMS使用的时候用的是回归前的置信度，这样就会导致一些回归后高IoU的预测框被删除。这就使得回归前的置信度并不能完全表征回归后的IoU大小。这样子也会导致算法精度的下降。在第一次使用NMS时候这种情况会比较明显，第二次使用的时候就会好很多，因此一阶段只使用一次NMS是会对精度有影响的，而二阶段目标检测中会在RPN之后进行一个更为精细的回归，在该处也会用到NMS，此时检测的精度就会好很多。

单阶段目标检测算法比两阶段算法相对来说效率更加高，原因是：单阶段目标检测算法相对于两阶段算法而言，具有更高的效率，主要是因为它在一个阶段中完成了两阶段算法中的两个步骤，即目标的定位和分类。两阶段算法通常首先使用选择性搜索算法对图像中的候选区域进行定位，然后使用分类器对每个候选区域进行分类。这两个步骤分别需要大量的计算资源，因此两阶段算法的效率通常相对较低。



## Faster-RCNN结构

1. Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
2. Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
3. Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
4. Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

### Region Proposal Networks(RPN)

​	经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如R-CNN使用 选择性搜索 SS(Selective Search)方法生成检测框。而Faster -RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster R-CNN的巨大优势，能极大提升检测框的生成速度。

### Bounding Box Regression边框回归

每个目标有两三个候选区域，每个候选区域都有目标概率值；原则上，1个物体对应1个候选区域，那么如何去除冗余的候选区域框，保留最好的1个？

![](https://img-blog.csdnimg.cn/6b3c4131bfad4fc58bed5a7c346d92f4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAa2luZ2E4OTg=,size_16,color_FFFFFF,t_70,g_se,x_16)

#### 非极大值抑制（Non-Maximum Suppression，NMS）

思路：选取那些邻域里分类数值最高，并且抑制那些分数低的窗口。

做法：设定阈值（阈值通常设定0.3~0.5 ），比较两两区域的IoU与阈值的关系。

Iou是两个区域的交并比：

![](https://img-blog.csdnimg.cn/ffd8726f92af45ceb2341326baa5f2e0.png)

那么可以如下两个思路来筛选候选框，假设阈值设定0.5：

（1）IoU>0.5，表示A框与B框重叠率高，可能是同一个物体，保留上一步计算的分类概率值高的候选框；
（2）IoU<0.5，表示A框与B框重叠率不高，可能是两个物体；

例子：假设检测到如下出6个都是人脸的矩形框，目的要找到最好的一个。

<img src="https://img-blog.csdnimg.cn/936f1866595e4028ae52044dafdd0bde.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAa2luZ2E4OTg=,size_19,color_FFFFFF,t_70,g_se,x_16" style="zoom:60%;" />

（1）根据分类器的类别分类概率做排序从小到大的概率分别为A、B、C、D、E、F；
（2）从最大概率矩形框F开始，分别判断A~E与F的重叠度IoU是否大于设定的阈值;
（3）假设B、D与F的重叠度超过阈值，那么就去除矩形框B、D；并标记第一个矩形框F，是我们保留下来的一个人脸框；
（4）从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，如果A、C的重叠度都大于设定的阈值，那么就去除；并标记E是我们保留下来的第二个矩形框；
（5）就这样一直重复，遍历所有，直到找到所有被保留下来的矩形框。

#### 对候选框坐标调整

如图所示绿色框为飞机的Ground Truth(GT)，红色为提取的positive anchors，即便红色的框被分类器识别为飞机，但是由于红色的框定位不准，这张图相当于没有正确的检测出飞机。所以我们希望采用一种方法对红色的框进行微调，使得positive anchors和GT更加接近。

![](https://pic4.zhimg.com/80/v2-93021a3c03d66456150efa1da95416d3_1440w.webp)

对于窗口一般使用四维向量 $(x,y,w,h)$ 表示，分别表示窗口的中心点坐标和宽高。红色框代表原始的positive anchors，绿色框代表目标的GT.目标。寻找一种关系，使得输入原始的anchor经过映射得到一个跟真实窗口G更接近的回归窗口G，即：

给定 anchor $A=(A_x,A_y,A_w,A_h)$ 和 $GT=(G_x,G_y,G_w,G_h)$，寻找一种变换 $f$，使得
$$
f(A_x,A_y,A_w,A_h) =(G_x,G_y,G_w,G_h)
$$
当predict与gt相差不大时候，变换f为平移缩放。



