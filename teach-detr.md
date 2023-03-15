## Introduction

经过近十年来经典探测器的快速发展，已经出现了许多成熟的设计和数十种成熟的基于rcnn的探测器，它们的性能都很有前景。我们想问以下问题:给定训练良好的基于rcnn的检测器和基于DETR的检测器，我们能否有效地转移它们的知识来训练一个?知识蒸馏(KD)[12]似乎是一种将知识从教师检测器转移到detr的可行方法。然而，我们在表1中的实验表明，现有的知识蒸馏方法[8,33,36]不能有效地将知识从基于rcnn的检测器转移到基于detr的检测器。

In this paper, we propose a novel training scheme, namely Teach-DETR, to construct proper supervision from teacher detectors for training more accurate and robust DETR models. We find that the predicted bounding boxes of teacher detectors can serve as an effective bridge to trans-fer knowledge between RCNN-based detectors and DETR-based detectors. We make the following important observa-tions: first, the bounding boxes are the unbiased representa-tion of objects for all detectors, which would not be affected by the discrepancies in detector frameworks, label assign-ment strategies, *etc*. Second, introducing more box anno-tations would greatly unleash the capacity of object queries by training them with various one-to-one assignment pat-terns [[3](#_bookmark31),[13](#_bookmark41)], and thus improve training efficiency and detec-tion performance. Finally, leveraging the predicted bound-ing boxes of teacher detectors would not introduce addi-tional architectures, since it has the similar format to the ground truth (GT) boxes. Nevertheless, it is non-trivial to integrate the GT annotations with the auxiliary supervi-sions. Due to the one-to-one matching of query-label as-signment of DETR-based detectors, incorporating the aux-iliary supervisions would be harmful to maintain the key signature of DETR of enabling inference without NMS. To tackle the challenge of transferring knowledge across dif-ferent types of detectors, we propose a solution that aligns with our observations and utilize output boxes and scores of the teacher detectors as extra *independent* and *parallel* su-pervisions for training the queries’ outputs. In addition, we employ the output scores of teacher detectors to measure the quality of the auxiliary supervisions and accordingly assign the predicted boxes different importance for weighting their losses. The ambiguity between GT boxes and the imperfect boxes from teachers can be alleviated and therefore enhance the importance of GT boxes.

Our Teach-DETR is a versatile training scheme and can be integrated with various popular DETR-based detectors without modifying their original architectures. Moreover, our framework has no requirement on teacher architectures, it can be RCNN-based detectors or DETR-based detectors, which is more general to various types of teachers. During training, our method introduces no additional parameters and adds negligible computational cost upon the original detector. During inference, our method brings zero addi-tional overhead. We conduct extensive experiments to prove that Teach-DETR can consistently improve the performance of DETR-based detectors. For example, our approach im-proves the state-of-the-art detector DINO [[35](#_bookmark63)] with Swin-Large [[21](#_bookmark49)] backbone, 4 scales of feature maps and 36-epoch training schedule, from 57.8% to 58.9% in terms of mean average precision on MSCOCO 2017 [[19](#_bookmark47)] validation set, demonstrating effectiveness of our method even when ap-plied to a state-of-the-art high-performance detector.

在本文中，我们提出了一种新的训练方案，即Teach-DETR，用于从教师检测器中构建适当的监督，以训练更准确和鲁棒的DETR模型。我们发现，教师检测器的预测边界盒可以作为基于rcnn检测器和基于detr检测器之间知识传递的有效桥梁。我们做了以下重要的观察:首先，包围框是所有检测器对象的无偏表示，它不会受到检测器框架、标签分配策略等差异的影响。其次，引入更多的框注释，可以通过各种一对一分配模式训练对象查询，从而极大地释放对象查询的能力[3,13]，从而提高训练效率和检测性能。最后，利用教师检测器的预测绑定盒不会引入额外的体系结构，因为它具有与ground truth (GT)盒类似的格式。然而，将GT注释与辅助监督集成起来并非易事。由于基于DETR的检测器的查询标签是一对一匹配的，引入辅助监督不利于保持DETR的关键签名，即在没有NMS的情况下启用推理。为了解决在不同类型的检测器之间传输知识的挑战，我们提出了一种与我们的观察相一致的解决方案，并利用输出框和大量教师检测器作为额外的独立和并行监督来训练查询的输出。此外，我们使用教师检测器的输出分数来衡量辅助监督的质量，并相应地赋予预测框不同的重要度来加权其损失。可以缓解GT盒子和老师不完善的盒子之间的模糊性，从而提高GT盒子的重要性。
我们的Teach-DETR是一种多功能的培训方案，可以与各种流行的基于detr的检测器集成，而无需修改它们的原始架构。此外，我们的框架对教师架构没有要求，它可以是基于rcnn的检测器，也可以是基于detr的检测器，对各种类型的教师更通用。在训练过程中，我们的方法没有引入额外的参数，并且在原始检测器上增加了可以忽略不计的计算成本。在推理过程中，我们的方法带来的额外开销为零。我们进行了大量的实验来证明Teach-DETR可以持续地提高基于detr的探测器的性能。例如，我们的方法改进了最先进的探测器DINO[35]，使用Swin-Large[21]主干，4个尺度的特征图和36 epoch训练计划，在MSCOCO 2017[19]验证集上的平均平均精度从57.8%提高到58.9%，证明了我们的方法即使应用于最先进的高性能探测器也是有效的。

## 相关工作

### DETR

DEtection TRansformer (DETR) [[1](#_bookmark29)] first applies Trans-formers [[30](#_bookmark58)] to object detection and achieves impressive performance without the requirement for non-maximum suppression. Despite its significant progress, the slow con-vergence of DETR is a critical issue which hinders its prac-tical implementation. There are many researchers devote to address the limitations and achieve promising speedups and better detection performance. Some works focus on design-ing better Transformer layers. For example, Deformable-DETR [[39](#_bookmark67)] introduces the multi-scale deformable attention scheme, which sparsely sampling a small set of key points referring to the reference points. Many works [[7](#_bookmark35), [20](#_bookmark48), [22](#_bookmark50), [32](#_bookmark60)] argue that the slow convergence of DETR mainly lies in that DETR could not rapidly focus on regions of interest. To ad-dress this issue, SMCA [[7](#_bookmark35)] and DAB-DETR [[20](#_bookmark48)] propose to modulate the cross-attentions of DETR, making queries to attend to restricted local regions. Anchor-DETR [[32](#_bookmark60)] ret-rospects the classical anchor-based methods and propose to use anchor points as object queries, one of which re-sponse for a restricted region. Conditional-DETR [[22](#_bookmark50)] de-couples each query into a content query and a positional query, which has a clear spatial meaning to limit the spatial range for the content queries to focus on the nearby region. Some recent approaches [[15](#_bookmark43), [35](#_bookmark63)] ascribe the slow conver-gence issue to the unstable bipartite matching [[14](#_bookmark42)]. They introduce an auxiliary query denoising task to stabilize bi-partite graph matching and accelerate model convergence. Two recent approaches [[3](#_bookmark31), [13](#_bookmark41)] look at this problem from a new perspective, arguing that slow convergence results from one-to-one matching. They propose to duplicate the GT boxes to support one-to-many matching in DETR. Unlike previous approaches, we propose a new training scheme to learn better DETR-based detectors from teachers. Theoret-ically, our approach is orthogonal to previous methods, and thus can further improve their performance.

检测变压器(DETR)[1]首次将变压器[30]应用于目标检测，在不需要非最大抑制的情况下取得了令人印象深刻的性能。尽管该方法取得了显著的进展，但收敛缓慢是阻碍其实际实施的关键问题。有许多研究人员致力于解决这些限制，并实现有前景的加速和更好的检测性能。一些工作专注于设计更好的Transformer层。例如，变形- detr[39]引入了多尺度变形注意方案，该方案参考参考点对一小组关键点进行稀疏抽样。许多著作[7,20,22,32]认为，DETR收敛缓慢主要是由于DETR不能快速聚焦感兴趣的区域。为了解决这一问题，SMCA[7]和DAB-DETR[20]提出对DETR的交叉注意进行调制，对受限的局部区域进行查询。锚点- detr[32]回顾了经典的基于锚点的方法，提出将锚点作为对象查询，其中一个锚点对一个受限区域进行响应。条件- detr[22]将每个查询解耦合为内容查询和位置查询，它具有明确的空间含义，以限制内容查询集中在附近区域的空间范围。最近的一些方法[15,35]将慢收敛问题归结为不稳定的二部匹配[14]。他们引入了一个辅助查询去噪任务来稳定双部图匹配和加速模型收敛。最近的两种方法[3,13]从新的角度看待这个问题，认为缓慢的收敛是一对一匹配的结果。他们建议复制GT盒以支持DETR中的一对多匹配。与以前的方法不同，我们提出了一种新的培训方案，从教师那里学习更好的基于detr的检测器。理论上，我们的方法与之前的方法是正交的，因此可以进一步提高它们的性能。

### Transfer Knowledge for Object Detection

For object detection, the commonly used way of lever-aging teacher detectors is knowledge distillation (KD) [[12](#_bookmark39)]. KD is usually applied to decrease the model complexity while improving the performance of the smaller student model. Compared to KD in classification, KD in object detection should consider more complex structure of boxes and somewhat cumbersome pipelines, and there are more hints, *e.g*., anchor predictions [[2](#_bookmark30), [26](#_bookmark54), [37](#_bookmark65)], proposal ranking results [[16](#_bookmark44)], object-related features [[4](#_bookmark32), [17](#_bookmark45), [31](#_bookmark59), [38](#_bookmark66)], contex-tual features [[8](#_bookmark36), [33](#_bookmark61)] and relations among features [[33](#_bookmark61), [34](#_bookmark62)], can be used. For DETR, there are few works have used KD. ViDT [[25](#_bookmark53)] try to utilize KD between output queries of teacher DETR and student DETR to reduce compution cost.In contrast, our method do not mean to conduct knowledge distillation or model compression. The core idea of Teach-DETR is to transfer knowledge of various teachers to train a more accurate and robust DETR-based detector. There-fore, the teacher detectors can be smaller or perform worse than the “student” DETR. Besides, our approach could be a meaningful exploration, providing a way to distill knowl-edge from any detectors to DETRs for future KD methods.

对于目标检测，常用的利用教师检测器的方法是知识蒸馏(KD)[12]。KD通常用于降低模型复杂度，同时提高较小的学生模型的性能。与分类中的KD相比，目标检测中的KD需要考虑更复杂的盒子结构和一些繁琐的管道，并且有更多的提示，如锚预测[2,26,37]，建议排序结果[16]，对象相关特征[4,17,31,38]，上下文-可以使用Tual特征[8,33]和特征之间的关系[33,34]。对于DETR，使用KD的作品很少。ViDT[25]尝试在教师DETR和学生DETR的输出查询之间利用KD来降低计算成本。相比之下，我们的方法并不意味着进行知识蒸馏或模型压缩。Teach-DETR的核心思想是通过传递不同教师的知识来训练一个更准确、更健壮的基于detr的检测器。因此，教师检测器可能比“学生”DETR更小或性能更差。此外，我们的方法可能是一次有意义的探索，为未来的KD方法提供了一种从任何探测器提取知识到detr的方法。



## Method

Our method aims to transfer knowledge of teacher detec-tors to train a more accurate and robust DETR-based detec-tor.  (i) We introduce the predicted bounding boxes of multi-ple teacher detectors, which can be RCNN-based detectors, DETR-based detectors or even both types of detectors, to serve as auxiliary supervisions for training DETR;  (ii) We weight the corresponding losses of the auxiliary supervi-sions with the predicted box scores from teacher detectors.  In this section, we will first retrospect DETR [1] (Sec. 3.1) and present details of Teach-DETR in (Sec. 3.2).

我们的方法旨在转移教师检测器的知识，以训练一个更准确、更健壮的基于detr的检测器。(i)引入多教师检测器的预测包围盒，可以是基于rcnn的检测器，也可以是基于DETR的检测器，甚至是两种检测器，作为训练DETR的辅助监督;(ii)我们将辅助监督的相应损失与教师检测器的预测框分进行加权。在本节中，我们将首先回顾DETR[1](第3.1节)，并在第3.2节中介绍Teach-DETR的细节。

### Training DETR with Teachers

**Exploration of DETR with knowledge distillation.** We aim at utilizing various types of teacher detectors to improve the detection performance of DETR-based detectors. A straightforward solution is to utilize knowledge distillation (KD) methods. However, existing KD methods show lim-ited effectiveness when conducting knowledge transfer between RCNN-based detectors and DETR-based detectors. We follow several state-of-the-art KD approaches [[8](#_bookmark36),[33](#_bookmark61),[36](#_bookmark64)] to perform feature imitation between the features of Mask RCNN’s FPN and the multi-scale features generated by Swin-Small backbone. In DeFeat [[8](#_bookmark36)], we also try to distill the classification logits[1](#_bookmark7) of the predicted boxes. As shown in Tab. [1](#_bookmark1), since there are much differences between the detec-tion pipelines of the RCNN-based and DETR-based detec-tors, it is very difficult to transfer knowledge from RCNN-based detectors to DETR-based ones. Existing KD methods even results in worse performance to the student detector.

基于知识蒸馏的DETR探索。我们的目标是利用各种类型的教师检测器来提高基于detr的检测器的检测性能。一个简单的解决方案是利用知识蒸馏(KD)方法。然而，现有的KD方法在基于rcnn的检测器和基于detr的检测器之间进行知识转移时，效果有限。我们采用几种最先进的KD方法[8,33,36]，在Mask RCNN的FPN的特征和Swin-Small主干生成的多尺度特征之间进行特征模仿。在DeFeat[8]中，我们还尝试提取预测框的分类logit1。如表1所示，由于基于rcnn的检测器和基于detr的检测器的检测管道存在很大差异，因此将知识从基于rcnn的检测器转移到基于detr的检测器是非常困难的。现有的KD方法甚至会导致学生检测器性能较差。

**Auxiliary supervisions from teacher detectors.**  In light of the above challenges, we propose to leverage the predicted bounding boxes of teachers as the auxiliary supervisions for better training DETRs.  There are two main reasons for using the predicted boxes for knowledge transfer.  First, the bounding box is the unbiased representation of results of all detectors, which would not be affected by the discrepancies in different detector frameworks, so it can serve as a good medium to transfer the knowledge of teachers to student DETRs.  Second, introducing more box annotations would largely unleash the capacity of object queries by providing more positive supervisions [3,13], and thus improve training efﬁciency and detection performance.  The whole pipeline of our Teach-DETR is shown in Fig. 1.

教师检测器的辅助监督。针对上述挑战，我们提出利用教师预测的边界框作为辅助监督，更好地训练detr。使用预测框进行知识转移有两个主要原因。首先，包围盒是所有检测器结果的无偏表示，不受不同检测器框架差异的影响，可以作为教师知识转移到学生DETRs的良好媒介。其次，引入更多的框注释，可以通过提供更多的正监督，在很大程度上释放对象查询的能力[3,13]，从而提高训练效率和检测性能。我们的Teach-DETR的整个管道如图1所示。

However, since the one-to-one matching is the critical design of DETRs to discard the NMS during inference, naively introducing the auxiliary supervisions would result in ambiguous training targets.  How to balance the contributions between different sets of teachers’ output boxes and GT boxes is a problem.  In Tab. 2, concatenating the auxiliary boxes and deteriorates the performance.

然而，由于一对一匹配是detr在推理时丢弃NMS的关键设计，因此天真地引入辅助监督会导致训练目标模糊。如何平衡不同教师输出盒和GT盒之间的贡献是一个问题。在表2中，连接辅助框会降低性能。

We try to address the above issue by conducting the one-to-one matching between GT boxes and object queries, and between each teacher's boxes and object queries independently. The matchings are also properly weighted according to the teachers' confidences on the predicted boxes. Specifically, for an input image, given K teacher detectors,we can obtain K sets of detection boxes, for the ith teacher, it contains for example M predicted bounding boxes,...,y}. Each of these predicted boxes y contains the estimated box size and location, the predicted category, and the predicted confidence scores. We apply one-to-one assignment between queries not only to the GT, but also to the auxiliary supervision of each teacher detector independently and thus we can obtain K groups of matching:

我们试图通过在GT盒子和对象查询之间进行一对一的匹配，以及在每个教师的盒子和对象查询之间独立进行匹配来解决上述问题。根据教师对预测框的置信度，对匹配进行适当加权。具体来说，对于一个输入图像，给定K个教师检测器，我们可以得到K组检测框，对于第i个教师，它包含例如M个预测边界框，…，y}。每个预测框y都包含估计的框大小和位置、预测的类别和预测的置信度分数。我们在查询之间不仅对GT进行一对一分配，而且对每个教师检测器的辅助监督进行独立分配，从而得到K组匹配: