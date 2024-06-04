# Numpy速成

参考文档：<a href="https://www.numpy.org.cn/user/quickstart.html#%E7%B4%A2%E5%BC%95%E3%80%81%E5%88%87%E7%89%87%E5%92%8C%E8%BF%AD%E4%BB%A3">Numpy中文网入门教程</a>

## 基础知识

NumPy的主要对象是同构多维数组。它是一个元素表（通常是数字），所有类型都相同，由非负整数元组索引。

NumPy的数组类被调用`ndarray`

`ndarray`对象更重要的属性是：

- **ndarray.ndim** - 数组的轴（维度）的个数。在Python世界中，维度的数量被称为rank。
- **ndarray.shape** - 数组的维度。这是一个整数的元组，表示每个维度中数组的大小。对于有 *n* 行和 *m* 列的矩阵，`shape` 将是 `(n,m)`。因此，`shape` 元组的长度就是rank或维度的个数 `ndim`。
- **ndarray.size** - 数组元素的总数。这等于 `shape` 的元素的乘积。
- **ndarray.dtype** - 一个描述数组中元素类型的对象。可以使用标准的Python类型创建或指定dtype。另外NumPy提供它自己的类型。例如numpy.int32、numpy.int16和numpy.float64。
- **ndarray.itemsize** - 数组中每个元素的字节大小。例如，元素为 `float64` 类型的数组的 `itemsize` 为8（=64/8），而 `complex32` 类型的数组的 `itemsize` 为4（=32/8）。它等于 `ndarray.dtype.itemsize` 。
- **ndarray.data** - 该缓冲区包含数组的实际元素。通常，我们不需要使用此属性，因为我们将使用索引访问数组中的元素。

### 数组的创建

```python
# 常见的错误是传入的不是列表
>>> a = np.array(1,2,3,4)    # WRONG
>>> a = np.array([1,2,3,4])  # RIGHT

# array方法可以将序列转化成列表
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
       [ 4. ,  5. ,  6. ]])

# 创建时显式指定数组的类型
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
       [ 3.+0.j,  4.+0.j]])

# zeros创建一个由0组成的数组，函数 ones创建一个完整的数组，函数empty 创建一个数组，其初始内容是随机的，取决于内存的状态。
# 默认情况下，创建的数组的dtype是 float64 类型的。
>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )
array([[6.23040373e-307, 8.45593934e-307, 7.56593017e-307],
       [1.95821439e-306, 2.22522597e-306, 8.45605478e-307]])
>>> np.empty( (2,3) ).dtype
dtype('float64')

# 按区间和步长创建数组
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
# 按区间和元素的个数创建数组（常用在创建浮点数数组中）
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
```

### 基本操作

数组上的算术运算符会应用到**元素**级别。

```python
# 逐元素运算
>>> a = np.array( [20,30,40,50] )
>>> b = np.arange( 4 )
>>> b
array([0, 1, 2, 3])
>>> c = a-b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
```

对于乘积运算符`*`在Numpy数组中是按元素相乘，矩阵乘积可通过`@`或者np.dot()

```python
>>> A = np.array( [[1,1],
...             [0,1]] )
>>> B = np.array( [[2,0],
...             [3,4]] )
>>> A * B                       # elementwise product
array([[2, 0],
       [0, 4]])
>>> A @ B                       # matrix product
array([[5, 4],
       [3, 4]])
>>> A.dot(B)                    # another matrix product
array([[5, 4],
       [3, 4]])
```

某些操作（例如`+=`和 `*=`）会更直接更改被操作的矩阵数组而不会创建新矩阵数组

```python
>>> a = np.ones((2,3), dtype=int)
# np的random模块的random方法
>>> b = np.random.random((2,3))
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += a
>>> b
array([[ 3.417022  ,  3.72032449,  3.00011437],
       [ 3.30233257,  3.14675589,  3.09233859]])
>>> a += b                  # b is not automatically converted to integer type
Traceback (most recent call last):
  ...
TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

当使用不同类型的数组进行操作时，结果数组的类型对应于更一般或更精确的数组（称为向上转换的行为）。

```python
>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0,pi,3)
>>> b.dtype.name
'float64'
>>> c = a+b
>>> c
array([ 1.        ,  2.57079633,  4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c*1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'
```

许多一元操作，例如计算数组中所有元素的总和，都是作为`ndarray`类的方法实现的。

```python
>>> a = np.random.random((2,3))
>>> a
array([[ 0.18626021,  0.34556073,  0.39676747],
       [ 0.53881673,  0.41919451,  0.6852195 ]])
>>> a.sum()
2.5718191614547998
>>> a.min()
0.1862602113776709
>>> a.max()
0.6852195003967595
```

默认情况下，这些操作适用于数组，就像它是一个数字列表一样，无论其形状如何。但是，通过指定`axis` 参数，您可以沿数组的指定轴应用操作：

```python
>>> b = np.arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

### 索引、切片、迭代

**多维**的数组每个轴可以有一个索引。这些索引以逗号分隔的元组给出：

```python
>>> def f(x,y):
...     return 10*x+y
...
>>> b = np.fromfunction(f,(5,4),dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2,3]
23
>>> b[0:5, 1]                       # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[ : ,1]                        # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, : ]                      # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])

# 当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片`:`
>>> b[-1]                                  # the last row. Equivalent to b[-1,:]
array([40, 41, 42, 43])
```

`b[i]` 方括号中的表达式 `i` 被视为后面紧跟着 `:` 的多个实例，用于表示剩余轴。NumPy也允许你使用三个点写为 `b[i,...]`。

三个点（ `...` ）表示产生完整索引元组所需的冒号。例如，如果 `x` 是rank为5的数组（即，它具有5个轴），则：

- `x[1,2,...]` 相当于 `x[1,2,:,:,:]`，
- `x[...,3]` 等效于 `x[:,:,:,:,3]`
- `x[4,...,5,:]` 等效于 `x[4,:,:,5,:]`。

```python
>>> c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
...                 [ 10, 12, 13]],
...                [[100,101,102],
...                 [110,112,113]]])
>>> c.shape
(2, 2, 3)
>>> c[1,...]                                   # same as c[1,:,:] or c[1]
array([[100, 101, 102],
       [110, 112, 113]])
>>> c[...,2]                                   # same as c[:,:,2]
array([[  2,  13],
       [102, 113]])
```

对多维数组进行 **迭代（Iterating）** 是相对于第一个轴完成的：

```python
>>> for row in b:
...     print(row)
...
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

但是，如果想要对数组中的每个元素执行操作，可以使用`flat`属性，该属性是数组的所有元素的迭代器：

```python
>>> for element in b.flat:
...     print(element)
...
```

## 形状操纵

以下三个命令都返回一个修改后的数组，但不会更改原始数组：

- `a.ravel()`
- `a.reshape()`
- `a.T`

```python
>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])
>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

而该 [`ndarray.resize`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize)方法会修改数组本身：

```python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])
```

### 堆叠数组

- `np.vstack()`：vertical
- `np.hstack()`：horizontal

```python
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

在复杂的情况下，[`r_`](https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_)和c [`c_`](https://numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_)于通过沿一个轴堆叠数字来创建数组很有用。它们允许使用范围操作符(“：”)。

```python
>>> np.r_[1:4,0,4]
array([1, 2, 3, 0, 4])
```

### 数组拆分

使用[`hsplit`open in new window](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit)，可以沿数组的水平轴拆分数组，方法是指定要返回的形状相等的数组的数量，或者指定应该在其之后进行分割的列：

```python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # Split a into 3
[array([[ 9.,  5.,  6.,  3.],
       [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
       [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
       [ 2.,  2.,  4.,  0.]])]
>>> np.hsplit(a,(3,4))   # 在第三列和第四列后切割
[array([[ 9.,  5.,  6.],
       [ 1.,  4.,  9.]]), array([[ 3.],
       [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
```

[`vsplit`](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit)沿垂直轴分割，并[`array_split`](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split)允许指定要分割的轴。

## 拷贝和视图

### 完全不复制

简单分配不会复制数组对象或其数据。

```python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```

### 视图或深拷贝

不同的数组对象可以共享相同的数据。该`view`方法创建一个查看相同数据的新数组对象。

1. **节省内存：** `view` 方法创建的新数组与原始数组共享数据，而不复制整个数组的内容。这意味着在某些情况下，你可以创建一个不同形状或数据类型的数组，而不需要额外的内存开销。
2. **更改形状：** 通过 `view`，你可以改变数组的形状，而不影响原始数组的形状。这是通过修改新数组的 `shape` 属性实现的。
3. **更改数据：** 如果你通过 `view` 修改新数组的元素，原始数组的数据也会相应地改变。这是因为它们共享相同的数据。

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

切片数组会返回一个视图：

```python
s = a[:, 1:3]   # 对 a 进行切片，获取一个视图 s，该视图包含 a 的所有行和列索引为 1 到 2 的列
s[:] = 10       # 将 s 中的所有元素设置为 10，由于 s 是 a 的视图，因此 a 的相应部分也被修改
a
# 输出:
# array([[   0,   10,   10,    3],
#        [1234,   10,   10,    7],
#        [   8,   10,   10,   11]])

```

### 深拷贝

该`copy`方法生成数组及其数据的完整副本。

```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

有时，如果不再需要原始数组，则应在切片后调用 `copy`。例如，假设a是一个巨大的中间结果，最终结果b只包含a的一小部分，那么在用切片构造b时应该做一个深拷贝：

```python
>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # the memory of ``a`` can be released.
```

如果改为使用 `b = a[:100]`，则 `a` 由 `b` 引用，并且即使执行 `del a` 也会在内存中持久存在。

## 广播机制

广播的规则确保参与运算的所有数组具有相同数量的维度，它的操作步骤如下：

1. 如果两个数组的维度数不同，那么在维度较小的数组的前面添加长度为1的维度，直到两个数组的维度数相同。
2. 在对齐的维度上，如果某个数组的大小为1，就将该维度进行复制，扩展成与另一个数组相同的大小。

```python
>>> import numpy as np
>>> a=np.array([1,2,3])
>>> b=np.array([[10],[20],[30]])
>>> result=a+b
>>> result
array([[11, 12, 13],
       [21, 22, 23],
       [31, 32, 33]])
>>>
```

## 索引技巧

### 使用索引数组进行索引

```python
>>> a=np.arange(12)**2
>>> a
array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121],
      dtype=int32)
>>> i=np.array([1,1,3,8,5])
>>> a[i]
array([ 1,  1,  9, 64, 25], dtype=int32)
>>> j=np.array([[3,4],[9,7]])
>>> a[j]
array([[ 9, 16],
       [81, 49]], dtype=int32)
>>>
```

索引数组是多维的情况

```python
>>> palette = np.array([
...     [0, 0, 0],      # 黑色
...     [255, 0, 0],    # 红色
...     [0, 255, 0],    # 绿色
...     [0, 0, 255],    # 蓝色
...     [255, 255, 255] # 白色
... ])
>>> image = np.array([
...     [0, 1, 2, 0],   # 每个值对应调色板中的一个颜色
...     [0, 3, 4, 0]
... ])
>>> result = palette[image]
>>> print(result)
[[[  0   0   0]
  [255   0   0]
  [  0 255   0]
  [  0   0   0]]

 [[  0   0   0]
  [  0   0 255]
  [255 255 255]
  [  0   0   0]]]
>>>
```

### 使用布尔数组进行索引

明确地选择我们想要的数组中的哪些项目以及我们不需要的项目。

最自然的布尔索引方法是使用与原始数组具有相同形状的布尔数组：

```python
>>> a = np.arange(12).reshape(3,4)
>>> b = a > 4
>>> b                                          # b is a boolean with a's shape
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
>>> a[b]                                       # 1d array with the selected elements
array([ 5,  6,  7,  8,  9, 10, 11])

# 此属性在属性分配中很有用
>>> a[b] = 0                                   # All elements of 'a' higher than 4 become 0
>>> a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

使用布尔值进行索引的第二种方法更类似于整数索引; 对于数组的每个维度，我们给出一个1D布尔数组，选择我们想要的切片：

```python
>>> a = np.arange(12).reshape(3,4)
>>> b1 = np.array([False,True,True])             # first dim selection
>>> b2 = np.array([True,False,True,False])       # second dim selection
>>>
>>> a[b1,:]                                   # selecting rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[b1]                                     # same thing
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[:,b2]                                   # selecting columns
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
>>>
>>> a[b1,b2]                                  # a weird thing to do
array([ 4, 10])
```

当涉及到 `ix_()` 函数时，它主要用于生成一个索引对象，该对象可以用于**获取多个输入数组的所有可能组合的值**。

在这个例子中，`np.ix_(a, b, c)` 返回了一个包含三个数组的元组。让我们逐步解释这个元组的每个部分：

1. 第一个数组 `result[0]` 是 `a` 的形状为 `(3, 1, 1)` 的三维数组，其中 `a` 的每个元素都被放在一个单元素的子数组中。	
2. 第二个数组 `result[1]` 是 `b` 的形状为 `(1, 3, 1)` 的三维数组，其中 `b` 的每个元素都被放在一个单元素的子数组中。
3. 第三个数组 `result[2]` 是 `c` 的形状为 `(1, 1, 3)` 的三维数组，其中 `c` 的每个元素都被放在一个单元素的子数组中。

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

result = np.ix_(a, b, c)
print(result)

# 输出（看不出来维度就去掉一个方括号看还有几个元素）
>>> (array([[[1]],
        [[2]],
        [[3]]]),
 array([[[4],
         [5],
         [6]]]),
 array([[[7, 8, 9]]]))

>>> ax,bx,cx=np.ix_(a,b,c)
>>> ax.shape
(3, 1, 1)
>>> bx.shape
(1, 3, 1)
>>> cx.shape
(1, 1, 3)

>>> result=ax+bx*cx
>>> result
array([[[29, 33, 37],
        [36, 41, 46],
        [43, 49, 55]],

       [[30, 34, 38],
        [37, 42, 47],
        [44, 50, 56]],

       [[31, 35, 39],
        [38, 43, 48],
        [45, 51, 57]]])

>>> result
array([[[29, 33, 37],
        [36, 41, 46],
        [43, 49, 55]],

       [[30, 34, 38],
        [37, 42, 47],
        [44, 50, 56]],

       [[31, 35, 39],
        [38, 43, 48],
        [45, 51, 57]]])
>>> result[0,1,2]
46
>>> a[0]+b[1]*c[2]
46
>>>
```

## 线性代数

```python
>>> import numpy as np
>>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> print(a)
[[ 1.  2.]
 [ 3.  4.]]

>>> a.transpose()
array([[ 1.,  3.],
       [ 2.,  4.]])

# 求矩阵的逆
>>> np.linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])

# E矩阵
>>> u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
>>> u
array([[ 1.,  0.],
       [ 0.,  1.]])
>>> j = np.array([[0.0, -1.0], [1.0, 0.0]])

# 矩阵乘法
>>> j @ j        # matrix product
array([[-1.,  0.],
       [ 0., -1.]])

# 矩阵的迹
>>> np.trace(u)  # trace
2.0

>>> y = np.array([[5.], [7.]])
# 求解线性方程组
>>> np.linalg.solve(a, y)
array([[-3.],
       [ 4.]])

# 求解特征值和特征向量
>>> np.linalg.eig(j)
(array([ 0.+1.j,  0.-1.j]), array([[ 0.70710678+0.j        ,  0.70710678-0.j        ],
       [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]]))
```

