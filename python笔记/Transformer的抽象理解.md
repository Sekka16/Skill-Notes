# Transformer的抽象理解

本文参考https://www.zhihu.com/question/362131975/answer/3297483297

## 1 Self-Attention机制

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Attention计算如上式，我们仅仅从数学角度对公式进行理解

### 1.1 矩阵乘积$QK^T$表示什么？

首先向量的内积有什么含义？

**向量的内积表征两个向量的夹角，亦表征一个向量在另一个向量上的投影。**所谓投影就是表明两个向量之间的相关性，并且投影的值越大则两个向量的相关性越高。

另外可以考虑，如果两个向量的夹角为直角，就说明这两个向量线性无关。

我们对于一个词进行编码，得到一个词向量，那么词向量的内积就是他们之间的相关性。

那么矩阵$QK^T$为一个方阵，里面保存每个词向量对于其他向量内积的结果，也就是每个词的相关性的值。

**这里也可以将相关性的值称为词A对词B的关注值的大小。**

### 1.2 Softmax

Softmax是一个将数值向量转化为概率分布的函数，将输入的原始数据转化成一个(0, 1)之间的述职进行归一化的操作。

在注意力机制上，Softmax的作用就是将向量与向量之间的相关性通过概率分布的形式表示出来。

那么为什么Softmax之后还要乘以V？

Softmax 的输出（注意力权重）只反映了序列中各个位置之间的相对关系，**它表示「每个查询向量应该关注输入的哪些部分」，但它本身没有包含输入的具体内容。** 

V表示输入序列中的内容信息，乘以注意力权重后，能够根据这些权重对内容进行加权求和，**输出不仅包含了注意力权重带来的关系特性，还能完整保留输入的内容特征。这是自注意力机制得以捕获全局依赖关系并生成有效上下文表示的关键。**

### 1.3 Q,K,V分别是什么？

Q是Query(查询)，K是Key(键)，V是Value(值)。在数据库中，我们希望通过一个查询Q去数据库中找出相应的值V，问题是直接通过Query去搜索数据库中的V往往效果不好，我们希望每一个V都对应一个K，这个K提取了V的特征，使得他更容易被找到。

因此Self-attention中的QKV，其实是想要**构建一个具有全局语义整合功能的数据库**，所以得到的最后矩阵就是输入矩阵X注意了上下文信息得到的语义矩阵。

### 1.4 $\sqrt{d_k}$是为了什么？

$QK^T$的分布方差会随着$\sqrt{QK^T}$增大而增大，输入范围过宽，Softmax 的指数运算会趋向于函数两端（0和1），最终输出的梯度几乎为零。这种梯度消失现象会导致模型难以训练，尤其是在深层网络中。

## 2 Transformer

Transformer主要由Encoder和Decoder两部分组成。

Encoder主要包含两层，一个是self-attention层一个前馈神经网络层，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。

Decoder也包含Encoder的self-attention层和前馈神经网络层，但中间增加了一层Encoder-Decoder Attention层，也叫cross-attention。

cross-attention与self-attention的区别是：**其Query(Q)来自Decoder，用于提出需求。Key(K)和Value(V)来自编码器，提供上下文信息。**

## 3 Self-attention与Transformer Encoder的实现

### 3.1 Self-attention

```python
import numpy as np
import torch
import torch.nn as nn

def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

def multihead_attention(X, mask, heads, W_KQV, W_out):
    B, T, d = X.shape
    K, Q, V = np.split(X @ W_KQV, 3, axis=-1)
    # shape of K, Q, V: B*T*d => B*heads*T*d/heads
    K, Q, V = [a.reshape(B, T, heads, d // heads).swapaxes(1, 2) for a in (K, Q, V)]
    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(d // heads) + mask)
    return (attn @ V).swapaxes(1, 2).reshape(B, T, d) @ W_out, attn

B, T, d = 50, 100, 64
X = torch.randn(B, T, d)
M = torch.triu(-float("inf")*torch.ones(T,T),1)

heads = 4
attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
Y_, A_ = attn(X,X,X,attn_mask=M)

Y, A = multihead_attention(X.numpy(), M.numpy(), heads,
                           attn.in_proj_weight.detach().numpy().T,
                           attn.out_proj.weight.detach().numpy().T)

print(np.linalg.norm(Y - Y_.detach().numpy()))
# 注意我们实现的multihead_attention与Pytorch的返回值有所不同，Pytorch返回的是多个头的均值
print(np.linalg.norm(A.mean(axis=1) - A_.detach().numpy())) 
```

### 3.2 Transformer Encoder

```python
import numpy as np
import torch
import torch.nn as nn

def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

def multihead_attention(X, mask, heads, W_KQV, W_out):
    B, T, d = X.shape
    K, Q, V = np.split(X @ W_KQV, 3, axis=-1)
    # shape of K, Q, V: B*T*d => B*heads*T*d/heads
    K, Q, V = [a.reshape(B, T, heads, d // heads).swapaxes(1, 2) for a in (K, Q, V)]
    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(d // heads) + mask)
    return (attn @ V).swapaxes(1, 2).reshape(B, T, d) @ W_out, attn

def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)

def relu(Z):
    return np.maximum(Z, 0)

def transformer(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
    Z = layer_norm(multihead_attention(X, mask, heads, W_KQV, W_out)[0] + X, eps)
    return layer_norm(Z + relu(Z @ W_ff1) @ W_ff2, eps)

B, T, d = 50, 100, 64
heads = 4
X = torch.randn(B, T, d)
M = torch.triu(-float("inf")*torch.ones(T,T),1)

trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
Y_ = trans(X, M)

Y = transformer(X.numpy(), M.numpy(), heads,
                trans.self_attn.in_proj_weight.detach().numpy().T, 
                trans.self_attn.out_proj.weight.detach().numpy().T,
                trans.linear1.weight.detach().numpy().T,
                trans.linear2.weight.detach().numpy().T,
                trans.norm1.eps)

print(np.linalg.norm(Y - Y_.detach().numpy()))
```