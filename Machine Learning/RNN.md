# Recurrent Neural Network

​	到目前为止我们默认数据都来自于某种分布，并且所有样本都是独立同分布的，然而，大多数的数据并非如此。例如，文章中的单词是按顺序写的，如果顺序被随机地重排，就很难理解文章原始的意思。同样，视频中的图像帧、对话中的音频信号以及网站上的浏览行为都是有顺序的。

​	如果说卷积神经网络可以有效地处理空间信息，那么RNN则可以更好地处理序列信息。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。

​	用 $x_t$表示t时间对应的值，$t\in \mathbf{Z}^+$为时间步(time step)。

## 隐藏变量自回归模型

​	是保留一些对过去观测的总结$h_t$，并且同时更新预测$\hat{x}_t$和总结$h_t$, 这就产生了 基于 $\hat{x}_t=P(x_t|h_t)$估计x_t，以及公式$h_t = g(h_{t−1}, x_{t−1})$更新的模型。由于$h_t$从未被观测到，因此被称为隐藏变量。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220925233919520.png" alt="image-20220925233919520" style="zoom:50%;" />



### 马尔可夫模型

马尔科夫模型( Markov model) 表示观测序列的联合概率分布：
$$
P(x_1,...,x_T)=\prod_{t=1}^TP(x_t|x_1,x_2,...,x_{t-1}) 当P(x_1|x_0) = P(x_1)
$$

#### 一阶马尔可夫链

观测 ${x_t}$的一阶马尔可夫链，其中，特定的观测 ${x_t}$的条件概率分布 ${P(x_t|x_{t-1})}$只以前一次观测 $x_{t-1}$为条件。
$$
P(x_1,...,x_T)=\prod_{t=1}^TP(x_t|x_{t-1}) 当P(x_1|x_0) = P(x_1)
$$




### 文本预处理

1. 将文本作为字符串加载到内存中。

2. 将字符串拆分为词元(如单词和字符)。

3. 建立一个词表，按出现频率排序，将拆分的词元映射到数字索引。 

4. 将文本转换为数字索引序列，方便模型操作。



### 语言模型和数据集

假设⻓度为$T$ 的文本序列中的词元依次为$x_1 , x_2 , . . . , x_T$ 。于是，$x_t (1 ≤ t ≤ T )$可以被认为是文本序列在时间步$t$处的观测或标签。在给定这样的文本序列时，语言模型(language model)的目标是估计序列的联合概率
$$
P(x_1,x_2,...,x_T)
$$



### 循环神经网络

#### 隐变量模型

​	n元语法模型，其中单词$x_t$在时间步$t$的条件概率仅取决于前面$n − 1$个单词。对于时间步$t − (n − 1)$之前的单词，如果我们想将其可能产生的影响合并到$x_t$上，需要增加$n$，然而模型参数的数量也会随之呈指数增⻓，因为词表V需要存储$|V|^n$个数字，因此与其将 $P(x_t | x_{t−1},...,x_{t−n+1})$ 模型化，不如使用隐变量模型:
$$
P(x_t | x_{t−1},...,x_{1}) = P(x_t |h_{t-1})
$$
​	$h_{t-1}$是隐状态，它存储了到时间步$t − 1$的序列信息，通常，我们可以基于当前输入$x_t$和先前隐状态$h_{t−1}$ 来计算时间步$t$处的任何时间的隐状态: 
$$
h_t = \phi(x_t,h_{t-1})
$$


#### 隐状态

​	假设我们在时间步$t$有小批量输入$X_t \in R^{n×d}$, 批量大小为n，输入维度为$d$。换言之，对于$n$个序列样本的小批量，**$X_t$的每一行对应于来自该序列的时间步$t$处的一个样本**。接下来，用$H_t \in R^{n×h}$ 表示时间步$t$的隐藏变量。与多层感知机不同的是，我们在这里保存了前一个时间步的隐藏变量$H_{t−1}$ ，并引入了一个新的权重参数$W_{hh} \in R_{h×h}$，来描述如何在当前时间步中使用前一个时间步的隐藏变量。**当前时间步隐变量由当前时间步的输入与前一个时间步的隐变量一起计算得出:**
$$
H_t = \phi{(X_tW_{xh} + H_{t-1}W_{hh} + \mathbf{b}_h )}
$$
​	输出层类似于多层感知机的计算：
$$
O_t = H_tW_{h\times q} + \mathbf{b}_q
$$




​	由于在当前时间步中，隐状态使用的定义与前一个时间步中使用的定义相同，因此 (7) 的计算是循环的(recurrent)。

​	图2为**按时间展开**的循环神经网络图：

<img src="https://img2018.cnblogs.com/blog/1630478/201904/1630478-20190414083004993-1402295824.png" style="zoom:80%;" />



#### 基于循环神经网络的字符级语言模型

​	语言模型目标是根据过去的和当前的词元预测下一个词元，因此将原始序列移位一个词元作为标签。如图为如何使用当前的字符和先前的字符预测下一个字符。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220924164809220.png" alt="image-20220924164809220" style="zoom:50%;" />



#### 模型的度量

**困惑度(Perplexity)**

​	通过一个序列中所有的$n$个词元的**交叉熵损失的平均值**来衡量，$x_t$ 是在时间步$t$从该序列中观察到的实际词元：
$$
L = \frac{1}{n}\sum_{t=1}^{n}-logP(x_t | x_{t−1},...,x_{1})
$$



#### 单隐藏层循环神经网络 pseudocode

```python
params: batch_size(批量大小), num_steps(时间步数), vocab(词表), train_iter(训练数据迭代器), epoch_num
weights: W_xh, W_hh, b_h, W_hq, b_q
整体数据输入形状 (时间步数量,批量大小,词表大小)
一个时间步内的输入X: 批量大小 x 词表大小
一个时间步内的隐状态: 批量大小 x 隐藏单元数.隐状态维度不变

function forward
	for X in inputs  # X:(batch_size,vocab_size) inputs:(num_steps, batch_size, vocab_size)
			H = tanh(matmul(X, W_xh) + matmul(H, W_hh) + b_h)
			Y = matmul(H, W_hq) + b_q
			output.append(Y) # 由隐状态h1,h2,...,ht输入到全连接层计算出了预测序列 y1,y2,...,yt
	return output

function train
		while i < epoch_num
			for X, Y in train_iter
          y_hat, state = net(X, state)
          l = loss(y_hat, y.long()).mean()
          updater.zero_grad()
          l.backward()
          updater.step()
	
```

### Encoder-Decoder架构

​	与上文中语料库是单一语言的语言模型问题存在不同，机器翻译的数据集是由源语言和目标语言的文本序列对组成的。机器翻译是序列转换模型的一个核心问题，其输入和输出都是⻓度可变的序列。

- Encoder-编码器：接受一个⻓度可变的序列作为输入，并将其转换为具有固定形状的编码状态；
- Decoder-解码器：将固定形状的编码状态映射到⻓度可变的序列。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20220926110931067.png" alt="image-20220926110931067" style="zoom:50%;" />



### 序列到序列学习(seq2seq)

​	独立的循环神经网络解码器是基于输入序列的编码信息和输出序列已经看⻅的或者生成的词元来预测下一个词元。现在使用**两个循环神经网络**实现序列到序列到学习。

[seq2seq](https://baijiahao.baidu.com/s?id=1650496167914890612&wfr=spider&for=pc)

#### Encoder

​	假设输入序列是$x_1, . . . , x_T$ ，其中$x_t$是输入文本序列中的第$t$个词元。在时间步$t$，循环神经网络将词元$x_t$的输入特征向量 $\mathbf{x}_t$和$h_{t−1}$(即上一时间步的隐状态)转换为$\mathbf{h}_t$(即 当前步的隐状态)。使用一个函数$f$ 来描述循环神经网络的循环层所做的变换:
$$
\mathbf{h}_t = f(\mathbf{x}_t,\mathbf{h}_{t-1})
$$
​	通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量。一般选择最后一个时刻t的隐状态，进行变换后得到序列的上下文变量。
$$
\mathbf{c} = q(\mathbf{h}_1,...,\mathbf{h}_T)
$$



<img src="https://pics6.baidu.com/feed/d833c895d143ad4be790326d30ec99aaa60f06d0.png@f_auto?token=ef10b1a74f448097ca1400d4a2ed9e30&s=1A247E2293E149031AE4657B02007072" style="zoom:60%;" />



#### Decoder

​	来自训练数据集的输出序列$y_1 , y_2 , . . . , y_{T′}$ ，对于每个时间步$t′$ (与输入序列或编码器的时间步$t$不同)，解码器输出$y_{t′}$ 的概率取决于先前的输出子序列 $y_1,...,y_{t′−1}$和上下文变量$\mathbf{c}$，即$P(y_{t′} | y_1,...,y_{t′−1},\mathbf{c})$。

​	使用另一个循环神经网络作为解码器。在输出序列上的任意时间步 $t′$，循环神经网络将来自上一时间步的输出$y_{t′−1}$ 和上下文变量 $\mathbf{c}$ 作为其输入，然后在当前时间步将它们和上一隐状态 $\mathbf{s}_{t′−1}$转换为隐状态$\mathbf{s}_{t′}$ 。因此，可以使用函数$g$来表示解码器的隐藏层的变换：

$$
\mathbf{s}_{t'} = g(y_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1})
$$
**Decoder的多种结构**

1. <img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221020001400131.png" alt="image-20221020001400131" style="zoom:20%;" />

   

   <img src="https://pics1.baidu.com/feed/faf2b2119313b07eea7eb46dbe39522695dd8c8c.png@f_auto?token=bffa7a38e8a3f03364843c4b487cc75c&s=A6B569220B9178C01261857902005071" style="zoom:50%;" />

​	将上下文变量**C**作为 RNN 的初始隐状态，输入到 RNN 中，后续只接受上一个时间步的隐状态 **h'** 而不接收其他的输入 **x**。



2. <img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221020001445643.png" alt="image-20221020001445643" style="zoom:20%;" />

   

   <img src="https://pics5.baidu.com/feed/267f9e2f0708283817c975a60a776a044e08f1b4.png@f_auto?token=969f4a8108c824804e0beefb840260c1&s=CCA43872D17BFDEF56DC017E0200E070" style="zoom:50%;" />

​	第二种 Decoder 结构有了自己的初始隐藏层状态 $h'_0$，不再把上下文变量 **c**当成是 RNN 的初始隐状态。



3. <img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20221020001619308.png" alt="image-20221020001619308" style="zoom:20%;" />



<img src="https://pics0.baidu.com/feed/203fb80e7bec54e74120bb580bd658554dc26a98.png@f_auto?token=43ca1d5d2006903ae68fc04f2c329651&s=4CA4387259DFF1E956D4017E0200A070" style="zoom:50%;" />



### 梯度裁剪

[Lipschitz连续](https://blog.csdn.net/ChaoFeiLi/article/details/110072841)



### 比较CNN、RNN、和 self-attention

​	**顺序操作无法并行化**	

下面几个架构目标是由n个词元组成的序列映射到另一个⻓度相等的序列，其中的每个输入词元或输出词元都由d维向量表示。考虑一个卷积核大小为k的卷积层，由于序列长度是n，输入和输出的通道数量都是d，所以卷积层的计算复杂度为 $O(knd^2)$。卷积神经网络是分层的，因此为有 $O(1)$ 个顺序操作，最大路径长度为 $O(n/k)$。例如，x1和x5处于图10.6.1中卷积核大小为3的双层卷积神经网络的感受野内。
当更新循环神经网络的隐状态时，dxd权重矩阵和d维隐状态的乘法计算复杂度为O（d2）。由于序列长度为n，因此循环神经网络层的计算复杂度为O（nd2）。根据图10.6.1，有O（n）个顺序操作无法并行化，最大路径长度也是O（n）。
在自注意力中，查询、键和值都是nxd矩阵。考虑（10.3.5）中缩放的”点一积“注意力，其中nxd矩阵乘以dxn矩阵。之后输出的nxn矩阵乘以nxd矩阵。因此，自注意力具有O（n2d）计算复杂性。正如我们在图10.6.1中看到的那样，每个词元都通过自注意力直接连接到任何其他词元。因此，有O（1）个顺序操作可以并行计算，最大路径长度也是O（1）。

<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20230227162601247.png" alt="image-20230227162601247" style="zoom:40%;" />
