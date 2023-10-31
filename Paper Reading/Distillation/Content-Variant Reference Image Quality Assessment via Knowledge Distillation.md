# Content-Variant Reference Image Quality Assessment via Knowledge Distillation



## Introduction

一般来说，人类更擅长感知高质量(HQ)和低质量(LQ)图像之间的差异，而不是直接判断单个LQ图像的质量。这种情况也适用于图像质量评价(IQA)。虽然最近的无参考(NR-IQA)方法在不依赖参考图像预测图像质量方面取得了很大的进展，但由于HQ图像信息没有被充分利用，它们仍然有潜力取得更好的性能。相比之下，全参考(full-reference, FR-IQA)方法倾向于提供更可靠的质量评价，但其实用性受到参考图像像素级对齐要求的影响。为了解决这个问题，本文首先提出了基于知识蒸馏的内容变体参考方法(content-variant reference method via knowledge distillation, CVRKD-IQA)。使用非对齐参考(NAR)图像来引入高质量图像的各种先验分布。对比HQ和LQ图像之间的分布差异，可以帮助我们的模型更好地评估图像质量。此外，知识蒸馏将更多的HQ-LQ分布差异信息从FR-teacher传递到NAR-student，稳定了CVRKD-IQA性能。为了充分挖掘局部-全局组合信息，同时实现更快的推理速度，该模型用MLP-mixer直接处理来自输入的多个图像块。跨数据集实验验证了该模型可以超越所有NAR/NR-IQA sota，甚至在某些情况下达到与FR-IQA方法相当的性能。由于内容变化和非对齐的参考HQ图像很容易获得，该模型可以以其对内容变化的相对鲁棒性支持更多的IQA应用。我们的代码和更详细的补充说明可在:https://github.com/guanghaoyin/CVRKD-IQA。

https://github.com/guanghaoyin/CVRKD-IQA.git

