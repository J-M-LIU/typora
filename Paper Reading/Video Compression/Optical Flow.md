# Optical Flow

## 传统方法

### Lucas/Kanade LK算法 (稀疏光流)

参考：Lucas/Kanade Meets Horn/Schunck: Combining Local and Global Optic Flow Methods

LK假设：

**物体运动变化较小**

**亮度不变假设**

假设待估计光流的两帧图像的同一物体的亮度不变，这个假设通常是成立的，因为环境光照通常不会发生太大的变化。I(x, y, t) 为t时刻位于(x, y)像素位置的灰度值。假设 t 时刻，位于 (x, y) 像素位置的物体，在 t+∆t时刻位于 (x+∆x, y+∆y) 位置，基于亮度不变假设：
$$
I(x,y,t) = I(x+\delta x,y+\delta y, t+ \delta t)
$$
泰勒公式展开可得：$I_x'$, $I_y'$分别为(x, y)像素点处图像亮度在x方向和y方向的梯度；∆I~t~为两图像之间(x, y)位置的亮度差。
$$
I(x+\delta x,y+\delta y, t+ \delta t) = I(x,y,t)+ \frac{\partial{I}}{\partial{x}} \delta x + \frac{\partial{I}}{\partial{y}} \delta y + \frac{\partial{I}}{\partial{t}} \delta t
\\I_x'u + I_y'v +I_t'\Delta t = I_x'u + I_y'v +\Delta I_t = 0
$$
可得
$$
\left[
\begin{matrix}
I_x'&I_y'
\end{matrix}
\right]
\left[
\begin{matrix}
u\\
v
\end{matrix}
\right]
= -\Delta I_t
$$
给定两张图像，$I_x'$, $I_y'$, $\Delta I_t$ 均为已知量，u,v 即为光流。存在两个未知 u, v 无法得到唯一解。因此，需要借助邻域光流相似假设。

**邻域光流相似假设**

以像素点 (x, y) 为中心，设定 n $\times$ n 的邻域，假设该邻域内所有像素点光流值一致。
$$
\left[
\begin{matrix}
I_x'^{(1)}&I_y'^{(1)} \\
I_x'^{(2)}&I_y'^{(2)} \\
...\\
I_x'^{(n)}&I_y'^{(n)} \\
\end{matrix}
\right]
\left[
\begin{matrix}
u\\
v
\end{matrix}
\right]
= 
\left[
\begin{matrix}
- \Delta I_t^{(1)} \\
- \Delta I_t^{(2)} \\
... \\
- \Delta I_t^{(n)}
\end{matrix}
\right]
$$
上式即为 Ax = b 形式，$A^TAX = A^Tb$. 最小二乘法求得 x = (u, v) 值。要求$A^TA$ 必须可逆。



### 金字塔LK

LK算法的约束条件：小速度，亮度不变以及区域一致性都是较强的假设，并不很容易得到满足。如当物体运动速度较快时，假设不成立，那么后续的假设就会有较大的偏差。

**思路**

考虑物体的运动速度较大时，算法会出现较大的误差。那么就希望能减少图像中物体的运动速度。一个直观的方法就是，缩小图像的尺寸。假设当图像为400×400时，物体速度为[16 16],那么图像缩小为200×200时，速度变为[8,8]。缩小为100*100时，速度减少到[4,4]。所以光流可以通过生成 原图像的金字塔图像，逐层求解，不断精确来求得。

**步骤**

1. 建立金字塔。原始图像位于底层，其上每一层均基于上一层进行计算。金字塔中每一层均是上一层的下采样，其计算公式为：
   $$
   I^L(x,y) = \frac{1}{4}I^{L-1}(2x,2y)+ \\
    \frac{1}{8}[I^{L-1}(2x-1,2y) + I^{L-1}(2x+1,2y)+ I^{L-1}(2x,2y-1) + I^{L-1}(2x,2y+1)] + \\
    \frac{1}{16}[I^{L-1}(2x-1,2y-1) + I^{L-1}(2x+1,2y+1)+ I^{L-1}(2x-1,2y-1) + I^{L-1}(2x+1,2y+1)] 
   $$



