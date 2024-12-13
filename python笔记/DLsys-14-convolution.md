# DLsys-14-convolution

## 1 存储顺序

### 1.1 图像的存储格式

对于图像，我们习惯于以`[N,H,W,C]`的格式进行存储，而PyTorch则是以`[N,C,H,W]`的格式进行存储的，尽管其在较新的版本中也支持前者。

存储的格式不同对性能常有实质性的差异。通常来说：`[N,H,W,C]`格式的卷积更快，因为他们可以更好的利用tensor core，但`[N,C,H,W]`格式对于BatchNorm操作通常更快，因为批归一化是对单个通道中的所有像素进行操作。

### 1.2 权重的存储格式

对于权重，我们习惯于以`[K,K,IN,OUT]`的格式进行存储，而PyTorch则是以`[OUT,IN,K,K]`的格式进行存储的。

**注意**，这里的`IN`对应的就是图像存储中的`C`，或许用`C_in`来表达更合适。

## 2 逐元素实现卷积

如前文所述，我们的图像将采用`[N,H,W,C]`的格式进行存储，权重将采用`[K,K,IN,OUT]`的格式进行存储。

为了验证我们的卷积实现是否正确，我们以PyTorch的实现作为参考。

```python
import torch
import torch.nn as nn

def conv_reference(Z, weight):
    # NHWC -> NCHW
    Z_torch = torch.tensor(Z).permute(0,3,1,2)
    
    # KKIO -> OIKK
    W_torch = torch.tensor(weight).permute(3,2,0,1)
    
    # run convolution
    out = nn.functional.conv2d(Z_torch, W_torch)
    
    # NCHW -> NHWC
    return out.permute(0,2,3,1).contiguous().numpy()

Z = np.random.randn(10,32,32,8)
W = np.random.randn(3,3,8,16)
out = conv_reference(Z,W)
print(out.shape)
```

好，现在我们以逐元素相乘累加来实现整个过程，思路也很简单，最内层的四重循环代表的是卷积核的滑动过程，中间两层代表图像的通道变化，最外层代表的图像的批次。

```python
def conv_naive(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    
    out = np.zeros((N,H-K+1,W-K+1,C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H-K+1):
                    for x in range(W-K+1):
                        for i in range(K):
                            for j in range(K):
                                out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i,j,c_in,c_out]
    return out

out2 = conv_naive(Z,W)
print(np.linalg.norm(out - out2))
```

## 3 矩阵乘法实现卷积

如果我们的卷积核的高和宽均为1，那么卷积实际上可以通过矩阵乘法`@`来实现。

假设输入Z为`[N,H,W,C]`，卷积核维度为`[K,K,IN,OUT]`，其中K=1，C=IN，那么输出特征图的维度为`[N,H+K-1,W+K-1,OUT]`，也即`[N,H,W,OUT]`。

对于这样的卷积，我们的计算过程可以是：

$$ Z[N,H,W,C_{in}] @ W[0,0], 其中Z.shape=[N,H,W,C_{in}],W[0,0].shape=[IN,OUT]$$

那么当卷积核的高度和宽度为K时，卷积的计算过程应该是什么样呢？

我们可以参考$1*1$的情况，每次只计算$K*K$中的一个点，然后再进行累加，即：

```python
def conv_matrix_mult(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    out = np.zeros((N,H-K+1,W-K+1,C_out))
    
    for i in range(K):
        for j in range(K):
            out += Z[:,i:i+H-K+1,j:j+W-K+1,:] @ weight[i,j]
    return out

Z = np.random.randn(100,32,32,8)
W = np.random.randn(3,3,8,16)

out = conv_reference(Z,W)
out2 = conv_matrix_mult(Z,W)
print(np.linalg.norm(out - out2))
```

事实上这样相当于把卷积核**对应位置点乘后再累加**的操作放在在最外层。

## 4 im2col

### 4.1 基本思想

在介绍`im2col`之前，我们先以一维卷积为例，假设我们需要处理这样一个卷积：

$$[0,x_1, x_2, x_3, x_4, x_5, 0] * [w_1, w_2, w_3]$$

为了提高效率充分并行，我们可以这样做：
$$
\begin{bmatrix}
0,x_1, x_2 \\
x_1,x_2,x_3 \\
x_2,x_3,x_4 \\
x_3,x_4,x_5 \\
x_4,x_5,0
\end{bmatrix}
\times
\begin{bmatrix}
w1\\
w2\\
w3
\end{bmatrix}
=
\begin{bmatrix}
x_1*w_2+x_2*w_3 \\
x_1*w_1+x_2*w_2+x_3*w_3\\
x_2*w_1+x_3*w_2+x_4*w_3\\
x_3*w_1+x_4*w_2+x_5*w_3\\
x_4*w_1+x_5*w_2\\
\end{bmatrix}
$$
这样我们在计算的时候，对于左边矩阵的每一行我们都可以并行执行。

**那么代价是什么呢？**

大量冗余的数据，正如公式中所示。空间效率和时间效率难以同时达到较好的效果。本质上来说，这是以空间换时间。

此外，我们通常将矩阵存储为二维数组，如A\[M\]\[N\]，在典型的优先的格式中，这会将矩阵的每个N维行一个接一个存储在内存中。然而这并不利于卷积计算中的内存访问，因为卷积核通常是K*K在图像上滑动，**运算会访问不连续的内存**。

为了有效提高内存访问，我们可以将其K*K访问到的数据放在连续的内存中，以A\[M/TILE\]\[N/TILE\]\[TILE\]\[TILE\]的形式，其中后两个维度即**连续的内存**。

### 4.2 存储形式转换

我们以一个6*6的矩阵为例，探讨如何将A\[M\]\[N\]转化成A\[M/TILE\]\[N/TILE\]\[TILE\]\[TILE\]

```python
import numpy as np
n = 6
A = np.arange(n**2, dtype=np.float32).reshape(n,n)
print(A)
```

```python
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10. 11.]
 [12. 13. 14. 15. 16. 17.]
 [18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29.]
 [30. 31. 32. 33. 34. 35.]]
```

这样的numpy矩阵，在内存上存储的形式是：

```python
import ctypes
print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
```

```bash
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
```

我们的目标是将其转化成2*2的小块，具体来说：第一块是[0,1,6,7]，第二块是[2,3,8,9]，以此类推。

为此，我们需要用到的是`np.lib.stride_tricks.as_strided()`方法，用于创建一个基于现有数组的视图（view），但通过调整步长（strides）来改变数组的形状和步幅。它不会复制数据，而是通过改变访问数据的方式来实现新的视图。

那我们就需要确定stride的四个维度，将A\[6\]\[6\]->B\[3\]\[3\]\[2\]\[2\]

第一维度：第一个tile是$\begin{bmatrix} 0,1\\6,7\end{bmatrix}$，y轴方向向下的下一个tile是$\begin{bmatrix} 12,13\\18,19\end{bmatrix}$，0与12在内存上的距离为12(48 bytes，每个int大小为4 bytes)，所以stride第一维为12

第二维度：第一个tile是$\begin{bmatrix} 0,1\\6,7\end{bmatrix}$，x轴方向向右的下一个tile是$\begin{bmatrix} 2,3\\8,9\end{bmatrix}$，0与2在距离为2，所以stride第二维度为2

第三维度：是tile内的维度，以$\begin{bmatrix} 0,1\\6,7\end{bmatrix}$为例，0与6的距离为6，所以stride第三维度为6

第四维度：是tile内的维度，以$\begin{bmatrix} 0,1\\6,7\end{bmatrix}$为例，0与1的距离为6，所以stride第四维度为1

综上所述，stride为\[12,2,6,1\]，同时由于int为4字节，我们需要在乘以4。

```python
B = np.lib.stride_tricks.as_strided(A, shape=(3,3,2,2), strides=np.array((12,2,6,1))*4)
print(B)
```

输出如下：

```bash
[[[[ 0.  1.]
   [ 6.  7.]]

  [[ 2.  3.]
   [ 8.  9.]]

  [[ 4.  5.]
   [10. 11.]]]

......

 [[[24. 25.]
   [30. 31.]]

  [[26. 27.]
   [32. 33.]]

  [[28. 29.]
   [34. 35.]]]]
```

此时仅仅改变了访问数据的方式，而真实的内存布局并没有受到影响，我们可以验证：

```python
print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=B.nbytes), B.dtype, B.size))
```

```bash
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
```

我们需要进一步调整其在内存中的布局，需要使用到的是`np.ascontiguousarray`

```python
C = np.ascontiguousarray(B)
print(np.frombuffer(ctypes.string_at(C.ctypes.data, size=C.nbytes), C.dtype, C.size))
```

```bash
[ 0.  1.  6.  7.  2.  3.  8.  9.  4.  5. 10. 11. 12. 13. 18. 19. 14. 15.
 20. 21. 16. 17. 22. 23. 24. 25. 30. 31. 26. 27. 32. 33. 28. 29. 34. 35.]
```

### 4.3 im2col实现卷积

什么是im2col，其全称为image to column，意思是将图像（或输入数据）转换为列的形式。

我们先以一个简单2D卷积为示例：

```python
A = np.arange(36, dtype=np.float32).reshape(6,6)
print(A)
```

```bash
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10. 11.]
 [12. 13. 14. 15. 16. 17.]
 [18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29.]
 [30. 31. 32. 33. 34. 35.]]
```

我们的卷积核为：

```python
W = np.arange(9, dtype=np.float32).reshape(3,3)
print(W)
```

```bash
[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
```

我们重新调整输入图像的布局，注意这里的B是一个视图，该视图所见元素具有大量重复，数量远远内存中实际存在的。

```python
B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
print(B)
```

```bash
[[[[ 0.  1.  2.]
   [ 6.  7.  8.]
   [12. 13. 14.]]

  [[ 1.  2.  3.]
   [ 7.  8.  9.]
   [13. 14. 15.]]

  ......

  [[20. 21. 22.]
   [26. 27. 28.]
   [32. 33. 34.]]

  [[21. 22. 23.]
   [27. 28. 29.]
   [33. 34. 35.]]]]
```

得到B之后，其内层两个维度即与卷积核对应的点乘累加操作，我们可以将**该部分与卷积核“拉直”，即im2col**，以**矩阵乘法**的形式完成一个2D的卷积。

```python
(B.reshape(16,9) @ W.reshape(9)).reshape(4,4)
```

```bash
array([[ 366.,  402.,  438.,  474.],
       [ 582.,  618.,  654.,  690.],
       [ 798.,  834.,  870.,  906.],
       [1014., 1050., 1086., 1122.]], dtype=float32)
```

值得一提的是，在实际优化中：

1. 避免实例化完整的 im2col 矩阵：

   - 在现代的高效实现中，通常不会直接生成完整的 im2col 矩阵，而是采用一种“懒惰”（lazy）的方式，即只在需要时动态生成部分数据。

   - 这种优化可以显著减少内存占用，同时保持计算效率。

2. 原生支持 im2col 矩阵的步长形式：

   - 一些高级的实现会直接针对 im2col 矩阵的步长形式（strided form）进行优化，而不是将其转换为标准的 2D 矩阵。

   - 这种优化可以避免内存分配的开销，同时利用硬件对步长访问的支持。

3. 快速分配和释放内存：
   - 在某些情况下，如果必须生成完整的 im2col 矩阵，可以采用快速分配和释放内存的策略，以减少内存占用的持续时间。例如，在卷积操作完成后立即释放 im2col 矩阵的内存

### 4.4 多通道小批量的im2col





