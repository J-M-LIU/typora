## Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer



### Introduction

基于Transformer多种可实现CV任务的模型应运而生，但由于参数过多导致的计算量剧增和内存占用过高问题，实际应用中迫切需要对transformer模型进行压缩。

**已有的压缩加速方法**

- compact network design  
- network pruning  
- low-rank decomposition
- quantization 
- knowledge distillation
- post-training quantization：PTQ——>模型性能降低
- QAT:quantization-aware training: higher comperssion rate without performance drop，对于CNN模型有效，但是还未探索一种适用于Vision Transformer的模型压缩方法



### Related Work



#### Vision Transformer

- DynamicViT
- Evo-ViT

以上模型着重于高效模型的设计，来得到更为轻量、快速的Vision Transformer；本文重点在于模型的**压缩**和**加速**，实现一个基于**QAT**的全量化Vision Transformer

#### Quantization

