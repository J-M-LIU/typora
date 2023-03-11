- quantization sensitivity
- data-free? zero-shot: what's the difference?



## Related Work

- QAT需要完整数据训练，对于生物医疗等难以获取的数据集，不太可行；且重新训练几百个epoches耗时费力；
- 关于PTQ的工作，如OMSE[^1] `优化原始张量和量化张量间的L2距离`，ACIQ[^2] `分析计算clip range`，然而这些都需要少量校准数据。
- DFQ, Data-Free Quantization[^3]，无训练数据和测试数据，使用分层量化。但当量化到6-bit和更低时，模型性能显著降低。
- 



## 参考文献

[^1]: Eli Kravchik, Fan Yang, Pavel Kisilev, and Yoni Choukroun. **Low-bit quantization of neural networks for efﬁcient inference.** In The IEEE International Conference on Computer Vision (ICCV) Workshops, Oct 2019.
[^2]: Ron Banner, Yury Nahshan, Elad Hoffer, and Daniel Soudry. **Post training 4-bit quantization of convolution networks for rapid-deployment.** CoRR, abs/1810.05723, 1(2), 2018.

[^3]: Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. **Data-free quantization through weight equalization and bias correction.** ICCV, 2019.