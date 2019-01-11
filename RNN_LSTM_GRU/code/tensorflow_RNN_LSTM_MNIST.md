# tensorflow_RNN_LSTM_MNIST with GPU kaggle

**Issue**: [tensorflow RNN kaggle__GPU](https://github.com/davidkorea/NLP_201811/issues/6) from 《Hands-On Machine Learning with Scikit-Learn&TensorFlow》 -  Chapter 14. Recurrent Neural Networks

<p align="center">
    <img src="https://i.loli.net/2019/01/06/5c31bee02bce4.png" width="500" height="300">
</p>

# 1. RNN with kaggle GPU
## 1.1 GPU
if you try to create each cell in a different
device() block, it will not work:
```python
with tf.device("/gpu:0"): # BAD! This is ignored.
layer1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
with tf.device("/gpu:1"): # BAD! Ignored again.
layer2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
```
This fails because a BasicRNNCell is a cell factory, not a cell per se (as mentioned earlier); no cells get
created when you create the factory, and thus no variables do either. The device block is simply ignored.
The cells actually get created later. When you call dynamic_rnn(), it calls the MultiRNNCell, which
calls each individual BasicRNNCell, which create the actual cells (including their variables).
Unfortunately, none of these classes provide any way to control the devices on which the variables get
created. If you try to put the dynamic_rnn() call within a device block, the whole RNN gets pinned to a
single device. So are you stuck? Fortunately not! The trick is to create your own cell wrapper:

```python
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
# import MNIST from tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist') # one_hot=False by default
# mnist = input_data.read_data_sets('/tmp/data/', one_hot=True) # dataset save path, 

x_test = mnist.test.images.reshape((-1,n_steps,n_inputs))
y_test = mnist.test.labels

# y_test = array([7, 2, 1, ..., 4, 5, 6], dtype=uint8) # one_hot=False
# one_hot=True
# array([[0., 0., 0., ..., 1., 0., 0.],
#        [0., 0., 1., ..., 0., 0., 0.],
#        [0., 1., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.]])
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

cells = DeviceCellWrapper("/gpu:0",tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)) # GPU
outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)

logits = fully_connected(inputs=states, num_outputs=n_outputs, activation_fn=None)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_operation = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits,y,1)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
```
```python
n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch = x_batch.reshape((-1,n_steps,n_inputs))
            sess.run(training_operation, feed_dict={x:x_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={x:x_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={x:x_test, y:y_test})
        print('{} epoch, Train accuracy: {:.3f} - Test accuracy: {:.3f}'.format(epoch, acc_train, acc_test))
```

tensorflow API **tf.fully_connected()** 
contains initialization of uniform weights and zero bias.

```
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
```
- **weights_initializer=initializers.xavier_initializer()**
```
tf.contrib.layers.xavier_initializer(
    uniform=True,
    seed=None,
    dtype=tf.float32
)
```
- **biases_initializer=tf.zeros_initializer()**

