# DLsys-14-convolution

## 1 存储顺序

### 1.1 图像的存储格式

对于图像，我们习惯于以`[N,H,W,C]`的格式进行存储，而PyTorch则是以`[N,C,H,W]`的格式进行存储的，尽管其在较新的版本中也支持前者。

存储的格式不同对性能常有实质性的差异。通常来说：`[N,H,W,C]`格式的卷积更快，因为他们可以更好的利用tensor core，但`[N,C,H,W]`格式对于BatchNorm操作通常更快，因为批归一化是对单个通道中的所有像素进行操作。

### 1.2 权重的存储格式

对于权重，我们习惯于以`[K,K,IN,OUT]`的格式进行存储，而PyTorch则是以`[OUT,IN,K,K]`的格式进行存储的。

注意，这里的`IN`对应的就是图像存储中的`C`，或许用`C_in`来表达更合适。

## 2 逐元素实现卷积

如前文所述，我们的图像将采用`[N,H,W,C]`的格式进行存储，我们的权重将采用`[K,K,IN,OUT]`的格式进行存储。

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

好，现在我们以简单的循环来实现整个过程，思路也很简单，最内层的四重循环代表的是卷积核的滑动过程，中间两层代表图像的通道变化，最外层代表的图像的批次。

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

假设我们的卷积核的高和宽均为1，那么卷积实际上可以通过矩阵乘法`@`来实现。

假设输入Z为`[N,H,W,C]`，卷积核维度为`[K,K,IN,OUT]`，其中K=1，C=IN，那么输出特征图的维度为`[N,H+K-1,W+K-1,OUT]`，也即`[N,H,W,OUT]`。

对于这样的卷积，我们的计算过程可以是：

$$ Z[N,H,W,C_{in}] @ W[0,0], 其中Z.shape=[N,H,W,C_{in},W[0,0].shape=[IN,OUT]$$

