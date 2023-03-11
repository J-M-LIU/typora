## 回顾Encoder-Decoder

假设输入序列是$x_1, . . . , x_T$ ，其中$x_t$是输入文本序列中的第$t$个词元。在时间步$t$，循环神经网络将词元$x_t$的输入特征向量 $\mathbf{x}_t$和$h_{t−1}$(即上一时间步的隐状态)转换为$\mathbf{h}_t$(即 当前步的隐状态)。使用一个函数$f$ 来描述循环神经网络的循环层所做的变换:
$$
\mathbf{h}_t = f(\mathbf{x}_t,\mathbf{h}_{t-1})
$$
通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量。一般选择最后一个时刻t的隐状态，进行变换后得到序列的上下文变量。
$$
\mathbf{c} = q(\mathbf{h}_1,...,\mathbf{h}_T)
$$

<img src="https://pics6.baidu.com/feed/d833c895d143ad4be790326d30ec99aaa60f06d0.png@f_auto?token=ef10b1a74f448097ca1400d4a2ed9e30&s=1A247E2293E149031AE4657B02007072" style="zoom:60%;" />

解码端因为语义编码 $\mathbf{c}$ 包含了整个输入序列的信息，所以在解码的每一步都引入 $\mathbf{c}$，作为t时刻的输入，每一时刻的语义编码 $\mathbf{c}$ 是相同的。
$$
\mathbf{y}_1 = g(\mathbf{c})\\
\mathbf{y}_2 = g(\mathbf{c},\mathbf{y}_1)\\
\mathbf{y}_3 = g(\mathbf{c},\mathbf{y}_1,\mathbf{y}_2)
$$
在生成目标句子的单词时，不论生成哪个单词，使用的语义编码C都相同。而语义编码 $\mathbf{c}$ 是由输入序列X的每个单词经过Encoder 编码产生的，这意味着输入序列X中任意单词对生成某个目标单词 $y_i$ 来说影响力都是相同的，没有任何区别（如果Encoder是RNN的话，理论上越是后输入的单词影响越大而并非等权，这也是Google提出Sequence to Sequence模型时发现把输入句子逆序输入做翻译效果会更好的小Trick的原因）

将整个序列的信息压缩在一个语义编码C中来记录整个序列的信息，序列较短还行，如果序列是长序列，那么只是用一个语义编码C来表示整个序列的信息会损失很多信息，且可能出现梯度消失问题。



<img src="https://img-blog.csdnimg.cn/20200321173021798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70" style="zoom:90%;" />

## 什么是attention

“是否包含自主性提示”将注意力机制与全连接层或汇聚层区别开来。在注意力机制的背景下，我们将自主性提示称为查询（query）。给定任何查询，注意力机制通过注意力汇聚（attention pooling）将选择引导至感官输入（sensory inputs，例如中间特征表示）。在注意力机制中，这些感官输入被称为值（value）。更通俗的解释，每个值都与一个键（key）配对，这可以想象为感官输入的非自主提示。我们可以设计注意力汇聚，以便给定的查询（自主性提示）可以与键（非自主性提示）进行匹配，这将引导得出最匹配的值（感官输入）。



## 注意力评分函数

假设有一个查询 $\mathbf{q}\in \mathbb{R}^{q}$ 和m个“key-value”对 $(\mathbf{k}_1,\mathbf{v}_1),...,(\mathbf{k}_m,\mathbf{v}_m)$，其中 $\mathbf{k}_i \in \mathbb{R}^{k}$ ，$\mathbf{v}_i \in \mathbb{R}^{v}$ ，注意力汇聚函数f 就被表示成值的加权和：
$$
f\left(\mathbf{q},\left(\mathbf{k}_{1}, \mathbf{v}_{1}\right), \ldots,\left(\mathbf{k}_{m}, \mathbf{v}_{m}\right)\right)=\sum_{i=1}^{m} \alpha\left(\mathbf{q}, \mathbf{k}_{i}\right) \mathbf{v}_{i} \in \mathbb{R}^{v}
$$
其中查询 $\mathbf{q}$ 和 键  $\mathbf{k}_i$ 的注意力权重是通过注意力评分函数 $s$ 将两个向量映射到标量，再经过 softmax运算得到：
$$
\alpha\left(\mathbf{q}, \mathbf{k}_{i}\right)=\operatorname{softmax}\left(s\left(\mathbf{q}, \mathbf{k}_{i}\right)\right)=\frac{\exp \left(s\left(\mathbf{q}, \mathbf{k}_{i}\right)\right)}{\sum_{j=1}^{m} \exp \left(s\left(\mathbf{q}, \mathbf{k}_{j}\right)\right)} \in \mathbb{R}
$$
选择不同的注意力评分函数s会导致不同的注意力汇聚操作.

阶段1：Query与每一个Key计算相似性得到相似性评分 s；
阶段2：将s评分进行softmax转换成[0,1]之间的概率分布；
阶段3：将[a1,a2,a3…an]作为权值矩阵对Value进行加权求和得到最后的Attention值。

<img src="https://img-blog.csdnimg.cn/20200322210257300.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70" style="zoom:90%;" />

