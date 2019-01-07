# tensorflow_RNN_LSTM_MNIST with GPU kaggle

**Issue**: [tensorflow RNN LSTM kaggle](https://github.com/davidkorea/NLP_201811/issues/6) from 《Hands-On Machine Learning with Scikit-Learn&TensorFlow》 -  Chapter 14. Recurrent Neural Networks

<p align="center">
    <img src="https://i.loli.net/2019/01/06/5c31bee02bce4.png" width="500" height="300">
</p>

# 1. RNN with kaggle GPU
## 1.1 GPU
```pytonn
class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device
    @property
    def state_size(self):
        return self._cell.state_size
    @property
    def output_size(self):
        return self._cell.output_size
    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)
```
## 1.2 RNN
```python
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
# tf.set_random_seed(1)
tf.reset_default_graph()
```
```python
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001
```
```python
x = tf.placeholder(tf.float32, (None, n_steps, n_inputs))
y = tf.placeholder(tf.int32, (None,))
```
