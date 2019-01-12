# tensorflow_Deep_RNN

<p align="center">
    <img src="https://i.loli.net/2019/01/12/5c39ada0dfd82.png" alt="Sample"  width="450" height="340">
</p>

# 1. MultiRNNCell
To implement a deep RNN in TensorFlow, you can create several cells and stack them into a MultiRNNCell. In the following code we stack three identical同样的 cells (but you could very well use various kinds of cells with a different number of neurons):

```python
n_neurons = 100
n_layers = 3
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
```

That’s all there is to it! The states variable is a tuple containing one tensor per layer, each representing the final state of that layer’s cell (with shape [batch_size, n_neurons]). If you set state_is_tuple=False when creating the MultiRNNCell, then states becomes a single tensor containing the states from every layer, concatenated along the column axis (i.e., its shape is [batch_size, n_layers * n_neurons]). Note that before TensorFlow 0.11.0, this behavior was the default.
