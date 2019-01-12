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
```python
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
          for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
```

That’s all there is to it! The states variable is a tuple containing one tensor per layer, each representing the final state of that layer’s cell (with shape [batch_size, n_neurons]). 

If you set state_is_tuple=False when creating the MultiRNNCell, then states becomes a single tensor containing the states from every layer, concatenated along the column axis (i.e., its shape is [batch_size, n_layers * n_neurons]). Note that before TensorFlow 0.11.0, this behavior was the default.

# 2. Dropout
```python
keep_prob = 0.5
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell_drop] * n_layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
```
Note that it is also possible to apply dropout to the outputs by setting output_keep_prob.
The main problem with this code is that it will apply dropout not only during training but also during
testing, which is not what you want (recall that dropout should be applied only during training).
Unfortunately, the DropoutWrapper does not support an is_training placeholder (yet?), so you must
either write your own dropout wrapper class, or have two different graphs: one for training, and the other
for testing. 
