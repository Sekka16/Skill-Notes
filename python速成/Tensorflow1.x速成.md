# Tensorflow1.x速成

项目来自：https://github.com/aymericdamien/TensorFlow-Examples/

## Introduction

### basic_eager_api：

```python
import numpy as np
import tensorflow as tf

# Set Eager API
print("Setting Eager mode...")
tf.enable_eager_execution()
tfe = tf.contrib.eager
```

Tensor数组和Numpy Arrays的混用

```python
# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

# 这里的matmul是矩阵乘法
d = tf.matmul(a, b)
print("a * b = %s" % d)
```

### basic_operations：

Tensorflow的静态图机制，简单来说先用描述性质的代码生成整体的计算图，后调用会话对计算图进行执行。

- `placeholder()`：创建计算图的时候为数据变量预留空间占位
- `tf.Session()`：创建会话
- `feed_dict`：为要计算的任务传参数

```python
# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
# Define some operations
add = tf.add(a, b)

# 这里的multiply就是简单的数字乘法
mul = tf.multiply(a, b)
# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print ("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print ("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
```

## Basic_Models

### logistics_regression

$$
Categorical\ Cross\ Entropy\ Loss =-\sum_{i=1}^C y_i \cdot \log \left(p_i\right)
$$

