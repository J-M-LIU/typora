# Rebuttal by Authors



**Global Response**

Dear Reviewers,

We deeply value the constructive feedback and insightful observations provided by all reviewers, and are delighted to see that the reviewers have acknowledged our effort toward xxxxx. We have exerted substantial effort to investigate and address all the issues raised. Should our clarifications meet the reviewers' expectations, we kindly request a reconsideration of the rating. We would like to sincerely appreciate all reviewers for their efforts again.





**Reviewer #4:**

**W1:** Sec 3.1: "\alpha is a predefined or learnable upper bound". In the actual implementation, is it predefined or learnable? If predefined, how were the values chosen? If learnable, what were the learnt values?

**A1:**

Sorry for the confusion. 在方法中，我们采用LSQ作为量化的基线方法，因此，$\alpha$ 为可学习的 scale parameter，as mentioned in [lsq]。 我们修改了此处的公式表达来避免confusion。We will rewrite our method description and correct the mathematical formulations in Sec 3.1.

Sorry for the confusion. In the actual implementation, we adapt LSQ as the quantization method. Therefore, $\alpha$ is learnable scale parameter as mentioned in[lsq]. We will rewrite our method description and correct the mathematical formulations in Sec 3.1





**Q2:** Is the bit-width selection for ROI/non-ROI regions is done on a per-frame basis, or on smaller regions? What if a frame has multiple ROI regions? Is the same bit-width selected for all ROI regions within a frame? How are the bit-widths used to encode a frame and MVs communicated to the decoder?



**A2:** 在我们的actual implementation中，bit-width selection是基于per frame的，通过UNISAL这种较为轻量的显著性识别网络，其耗时相较于编解码可以忽略不计。针对一帧内的multiple regions，bit-width的选择是相同的，是基于区域的整体平均复杂度来计算。关于每一帧的bit-widths，实验中直接将其写入码流中的自定义数据部分（SEI），并发送到解码端，解码部分可直接解析码流中的bit-widths数据。

bit-width的选择是针对每帧进行的，基于区域的平均复杂度来选择。因此假如一帧内有多个ROI 区域，bir-width selection是相同的。对于每一帧的bit-width选择结果，我们使用2个16 bit integer 进行旁路编码，将其直接写进二进制码流。



**Q3:** Specifics on the calculation of $\sigma(x),$ $\sigma(v)$, G(x), G(v) in Eq. (5) should be provided. Are these calculated for each pixel within each region, or a single value calculated for each region? What are the window sizes used to calculate these values? These parameter choices could have a significant impact on the actual performance.



**A3:** Sorry for the confusion in the formula. 在我们的具体实现中， 我们分别计算ROI/non-ROI区域的整体的 standard deviation value 和 average magnitude，这个值针对每像素计算。之后我们将这几个metrics 进行concatenate后，送入一个全连接层 fc: $R^{8} \Rightarrow R^{K}$, 其中 K代表 每个区域的K个 bit candidates。感谢您提出的问题，我们在revision中对这一部分进行更细致地修改。

C:3+3+2+2 -> K



补充在各个数据集上的统计表：人像、大运动、blabla等的qualitative results

**Q4:** No comparison against standard codecs such as HEVC and/or VVC is provided.





**A4:**



**Q5:** For the results in Section 4.2, the authors should provide values for the BD-BR metric which is the standard way of evaluating and comparing codec performance.



**A5:**



**Q6:** The evaluation approach for complexity is not very clear. How are the FLOP values calculated for different bit-widths? For instance, is it assumed that eight 4-bit operations is equivalent to 1 FLOP? How would this dynamic quantization be implemented on an actual GPU? Are the values actually converted to fp8, fp4, or int8, int4, etc.? What about values like 5-bit or 6-bit as indicated in the results in Fig. 5? There is no native support for such bit-widths in a GPU, and so an actual implementation will likely require representing in fp32 (even though the value itself might be quantized to a lower number of bits) which wouldn't result in any speed up in operation. Hence, I would suggest that the authors report not just flop counts, but also improvement in actual run-times to support their claims of efficiency. Also, please provide details on "ROI and non-ROI regions are allocated different computing resources."



**A6:** 感谢您提出的问题，在复杂度的衡量上我们未给出详细的定义。具体而言，我们采用Bit-FlOPs来表示计算复杂度。~~Bit-FLOPs are determined by multiplying the bit-width of the weights, the bit-width of the activations, and the FLOPs, which consist of multiplication and addition operations.~~

As the bitwise XNOR operation and bit-counting can be performed in a parallel of 64 by the current generation of CPUs, the FLOPs is calculated as the amount of real-valued floating point multiplication plus 1/64 of the amount of 1-bit multiplication.

$\frac{||b_a \cdot b_w ||_2}{32} * 2 * C_{in} * C_{out} * F^2 * B * H * W$

Following idea of [Q-DETR], CPUs are capable of performing both bit-wise XNOR and bit-count operations concurrently. The calculation results equals the Bit-OPs following [Zechun Liu, Wenhan Luo, Baoyuan Wu, Xin Yang, Wei Liu, **952** and Kwang-Ting Cheng. Bi-real net: Binarizing deep net- **953** work towards real-network performance. *International Journal of Computer Vision*, 128(1):202–219, 2020 ]



We follow [42] to calculate memory usage, which is estimated by adding 32× the number of real-valued weights and a× the number of quantized weights in the a-bit networks. The number of operations (OPs) is calculated in the same way as [42]. The current CPUs can handle both bit-wise XNOR and bit-count operations in parallel. The respective number of real-valued FLOPs plus { 1 , 1 , 1 } 32 16 8 of the number of {2,3,4}-bit multiplications equals the OPs following [24].



对于不同区域的特征，FLOPs计算：xxxxx

关于设计推理时间的测试，我们进一步补充了 由于目前没有对于 5-bit，6-bit的原生支持，我们将4-bit 和 8-bit candidates设置为 {2,4,8} 和 {4，8，16}。 On average，



