# MultiHead_Attention与TransformerEncoderLayer

## MultiHead_Attention

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
print(np.linalg.norm(A.mean(axis=1) - A_.detach().numpy()))
```

> 注意：这里我们的实现与pytorch有所不同，pytorch返回的多个head的平均值

## TransformerEncoderLayer

```python
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

