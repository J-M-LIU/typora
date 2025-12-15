$$
Q_{b^k}(\boldsymbol{x})=\lfloor\operatorname{clip}(\boldsymbol{x}\cdot\frac{s(b^k)}{\alpha^k}, -Q_N^{{b^k}}, Q_P^{{b^k}})\rceil \cdot \frac{\alpha^k}{s(b^k)},
$$







We run the widely used, non-neural, standard codec HEVC [31] (*a.k.a.* H.265) using the ffmpeg x265 codec in *veryslow* settings, as well as H.264 using x264 in the *medium* setting



我们也在标准编码 H.265/HEVC 和 H.264/AVC上对各个数据集进行了测试，采用ffmpeg x265 and x264 in *veryslow* mode.

We also run standard codecs H.265 (HEVC) and H.264 (AVC) on test datasets, using ffmpeg x265 and x264 in *veryslow* mode.



ROI区域的平均量化bit-width高于non-ROI区域，因此分配更多计算资源。

在运动较为big的HEVC-D数据集上，bit-allocator更倾向于选择higher bit-width；而对于运动较小且

xxx的占比更大；而在相对运动较小的UVG和HEVC-E数据集上，lower bit-widths则占比较多。

For HEVC-D datasets with relatively large motion, more ROI/non-ROI regions use higher bit-width. On the UVG and HEVC-E datasets with relatively small motion, lower bit-widths account for more.



在具有复杂前景及单调背景的HEVC-E数据集上，ROI区域中，47%的帧被分配10-bit；而xxx%的frames中non-ROI区域被分配更低的6-bit；在纹理复杂且运动剧烈的HEVC-C数据集上，ROI区域和non-ROI区域中higher bit-width均占比较高。

ROI和non-ROI区域均分配更高的bit-width

gholami2022survey

habibian2019video