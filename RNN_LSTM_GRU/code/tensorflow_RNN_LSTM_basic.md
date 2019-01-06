
# tensorflow_RNN_LSTM_basic
https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_RNN_LSTM_basic.ipynb


(https://i.loli.net/2019/01/04/5c2f0a2a35b04.png)

与[numpy RNN tutorail](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_RNN.md)不同, 由于此处不需要生成文本text generate，所以hidden计算后，无需进行softmax。所以hidden直接输出为y。

因此，此处2 timestep的例子中，将前面的输出y直接传递给后面来计算，原因就是此处hidden就是y

output_seqs, states = tf.contrib.rnn.static_rnn(cell=basic_cell, inputs=[x0,x1], dtype=tf.float32)
  - output_seqs：每个timestep的输出结果
  - states：final states of the network，应该就是hidden吧？就是memory？


-----

# 1. Basic RNNs in TensorFlow


<p align="center">
    <img src="https://camo.githubusercontent.com/11597f39a693e699966f443bd2d15eb5bef45f87/68747470733a2f2f692e6c6f6c692e6e65742f323031382f31322f31382f356331383530623235353366632e706e67">
</p>

<p align="center">
    <img src="https://camo.githubusercontent.com/a7d62cc5a1d4412efc7153d13e7601d8881681d5/68747470733a2f2f692e6c6f6c692e6e65742f323031382f31322f31382f356331383466643135336637382e706e67">
</p>

We will assume that the RNN runs over only two time steps, taking input vectors of size 3 at each time step. The following code builds this RNN, unrolled through two time steps:

1. time steps : 2, two words a sentence
2. RNN cell input vector dim: 3, each RNN cell accepts a shape of (1,3) vector
3. hidden size: 5

```python
import tensorflow as tf
import numpy as np
tf.set_random_seed(1)
```

```python
n_inputs = 3 # input vector dim at each time step 
n_neurons = 5 # RNN cell hidden size
```

```python
# 2 time steps: x0, x1
x0 = tf.placeholder(tf.float32, shape=(None,n_inputs))
x1 = tf.placeholder(tf.float32, shape=(None,n_inputs))
```

```python
wx = tf.Variable(tf.random_normal(shape=(n_inputs, n_neurons), dtype=tf.float32))
wy = tf.Variable(tf.random_normal(shape=(n_neurons, n_neurons), dtype=tf.float32))
b = tf. Variable(tf.zeros(shape=(1,n_neurons), dtype=tf.float32))

y0 = tf.tanh(tf.matmul(x0,wx)+b) # shape (1,3)x(3,5)+(1,5) = (1,5)
y1 = tf.tanh(tf.matmul(y0,wy)+tf.matmul(x1,wx)+b)

init = tf.global_variables_initializer()
```

```python
# mini batch = 4,     sentence1   sentence2  sentence3  sentence4
x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t=0
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t=1

# sentence1: x0 = [0, 1, 2], x1 = [9, 8, 7]
# sentence2: x0 = [3, 4, 5], x1 = [0, 0, 0]
# sentence3: x0 = [6, 7, 8], x1 = [6, 5, 4]
# sentence4: x0 = [9, 0, 1], x1 = [3, 2, 1]
```



