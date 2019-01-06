
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
### 仅根据RNN数学原理，未使用tensorflow自带RNN函数

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
n_inputs = 3 # input vector dim at each time step, 每个timestep输入单词的向量长度
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
b = tf.Variable(tf.zeros(shape=(1,n_neurons), dtype=tf.float32))

y0 = tf.tanh(tf.matmul(x0,wx)+b) # shape (1,3)x(3,5)+(1,5) = (1,5)
y1 = tf.tanh(tf.matmul(y0,wy)+tf.matmul(x1,wx)+b)

init = tf.global_variables_initializer()
```

```python
# mini batch = 4,    sentence1   sentence2  sentence3  sentence4
x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t=0
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t=1

# sentence1: x0 = [0, 1, 2], x1 = [9, 8, 7]
# sentence2: x0 = [3, 4, 5], x1 = [0, 0, 0]
# sentence3: x0 = [6, 7, 8], x1 = [6, 5, 4]
# sentence4: x0 = [9, 0, 1], x1 = [3, 2, 1]
```
```python
with tf.Session() as sess:
    sess.run(init)
    y0_val, y1_val = sess.run([y0,y1], feed_dict={x0:x0_batch, x1:x1_batch})
```
```python
print('y0_val:\n',y0_val)
print('\ny1_val:\n',y1_val)


y0_val:
 [[-0.91210926 -0.97909343 -0.9963769  -0.804197    0.81554604]
 [-0.9984175  -0.99501723 -0.9999899  -0.999553    0.878944  ]
 [-0.99997276 -0.99881977 -1.         -0.99999905  0.921494  ]
 [ 0.9867835   1.          1.         -0.5567072  -0.9989798 ]]

y1_val:
 [[-0.9999856   0.8339144  -0.9999802  -0.99999195 -0.97266006]
 [-0.9276461   0.8301138   0.03124463  0.9562516  -0.9378031 ]
 [-0.9995392   0.97095305 -0.99141777 -0.99612236 -0.9843221 ]
 [-0.5459391  -0.3920396   0.943575   -0.99999213  0.9344761 ]]
```


Each timestep output of sentence 1
```
y0 = [-0.9157372   0.9512312  -0.8827221  -0.9911751  -0.9362729 ]
y1 = [ 0.99998605  1.          1.          0.99998194  0.93795544]
```
Each timestep output of sentence 2
```
y0 = [ 0.8368062   0.99866116  0.29276526 -0.9999999  -0.99998057]
y1 = [-0.97918355 -0.99938613  0.9646454  -0.9705091  -0.9811791 ]
```
Each timestep output of sentence 3
```
y0 = [ 0.9993057   0.9999641   0.96339613 -1.         -1.        ]
y1 = [ 0.9958529   1.          0.9999868   0.9993763   0.9095211 ]
```

Each timestep output of sentence 4
```
y0 = [ 1.         -0.99971396  0.9999998  -0.9999982  -0.97780967]
y1 = [ 1.          0.17404278  1.         -1.         -1.        ]
```

Each output of one timestep is the size of (1, hidden_size) = (1, 5)

That wasn’t too hard, but of course if you want to be able to run an RNN over 100 time steps, the graph is going to be pretty big. Now let’s look at how to create the same model using TensorFlow’s RNN operations.


# 2. Static Unrolling Through Time
## 2.1 使用tensorflow自带BasicRNNCell，static_rnn

```python
tf.reset_default_graph() # RESET THE DEFAULT GRAPH USED ABOVE

n_inputs = 3 # input vector dim at each time step 
n_neurons = 5 # RNN cell hidden size

x0 = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs))
x1 = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(cell=basic_cell, inputs=[x0,x1], dtype=tf.float32)
y0,y1 = output_seqs
```

First we create the input placeholders, as before. Then we create a ```BasicRNNCell()```, which you can think
of as a factory that creates copies of the cell to build the unrolled RNN (one for each time step). 

Then we call ```static_rnn()```, giving it the cell factory and the input tensors, and telling it the data type of the inputs(this is used to create the **initial state matrix, which by default is full of zeros**). 

The ```static_rnn()```function calls the cell factory’s``` __call__()``` function once per input, creating two copies of the cell (each containing a layer of five recurrent neurons), **with shared weights and bias terms**, and it chains them just like we did earlier. 

```tf.contrib.rnn.BasicRNNCell(num_units, activation=None, reuse=None, name=None, dtype=None, **kwargs)```

```tf.contrib.rnn.static_rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)```

The ```static_rnn()``` function returns two objects. 
1. The first is a Python list containing the output tensors for each time step. 
2. The second is a tensor containing the final states(hidden states) of the network. When you are using basic cells, the final state is simply equal to the last output. show as the code printed below.


```python
# mini batch = 4,     sentence1   sentence2  sentence3  sentence4
x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t=0
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t=1
with tf.Session() as sess:
    # sess.run(init)
    sess.run(tf.global_variables_initializer())
    y0_val, y1_val, states = sess.run([y0,y1,states], feed_dict={x0:x0_batch, x1:x1_batch})
```
```python
print('y0_val:\n',y0_val)
print('\ny1_val:\n',y1_val)
print('\nstates:\n',states)



y0_val:
 [[ 0.65068823 -0.7885722  -0.68872046 -0.579271    0.5519878 ]
 [ 0.9604823  -0.9950424  -0.9997227  -0.5949103   0.8591207 ]
 [ 0.9961673  -0.9998956  -0.9999998  -0.6101117   0.96099204]
 [-0.4338037  -0.33560485 -0.9999477  -0.36008093  0.70006984]]

y1_val:
 [[ 0.9983586  -0.9998548  -1.          0.80598366  0.7580543 ]
 [ 0.8172029   0.09068511 -0.8805279   0.5421987  -0.5194732 ]
 [ 0.9914208  -0.99018264 -0.99999976  0.845585    0.12200028]
 [ 0.6654841  -0.707808   -0.99619687  0.8441219  -0.2600849 ]]
 
states:
 [[ 0.9983586  -0.9998548  -1.          0.80598366  0.7580543 ]
 [ 0.8172029   0.09068511 -0.8805279   0.5421987  -0.5194732 ]
 [ 0.9914208  -0.99018264 -0.99999976  0.845585    0.12200028]
 [ 0.6654841  -0.707808   -0.99619687  0.8441219  -0.2600849 ]]
```
## 2.2 简化输入x

If there were 50 time steps, it would not be very convenient to have to define 50 input placeholders and 50 output tensors.

x0 = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs))
x1 = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs))
...

x49 = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs))
Moreover, at execution time you would have to feed each of the 50 placeholders and manipulate the 50 outputs.

Let’s simplify this. The following code builds the same RNN again, but this time it takes a single input placeholder of shape [batch_size, n_steps, n_inputs] where the first dimension is the mini-batch size. 즉, put all 50 timesteps (50 words a sentence) together in one tenser/array/list.

```python
tf.reset_default_graph()

n_steps = 2

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

x_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, 
                                                x_seqs, 
                                                dtype=tf.float32)
                                                
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
```

input placeholder of shape [None, n_steps, n_inputs] where the first dimension is the mini-batch size.

Then it extracts the list of input sequences for each time step. X_seqs is a Python list of n_steps tensors of shape [None, n_inputs], where once again the first dimension is the minibatch size.

To do this, we first swap交换 the first two dimensions using the transpose() function, so that the time steps are now the first dimension.

**此处为什么要交换tensor的前两个维度，将timestep放在第一个维度。是因为static_RNN()只支持第一维度为timestep的tensor吗？？？？？**

Then we extract取出 a Python list of tensors along the first dimension (i.e., one tensor per time step) using the unstack() function.

The next two lines are the same as before. Finally, we merge all the output tensors into a single tensor using the stack() function, and we swap the first two dimensions to get a final outputs tensor of shape [None, n_steps, n_neurons] (again the first dimension is the mini-batch size).

```python
x_batch = np.array([
                    #  t = 0      t = 1
                    [[0, 1, 2], [9, 8, 7]], # instance 0
                    [[3, 4, 5], [0, 0, 0]], # instance 1
                    [[6, 7, 8], [6, 5, 4]], # instance 2
                    [[9, 0, 1], [3, 2, 1]], # instance 3
                    ])
```
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_val,states = sess.run([outputs, states], feed_dict={x:x_batch})
    # x = (None, n_steps, n_inputs)
    # x_batch = (4, 2, 3) batch=4, timestep=2, input_vector=3
```
```python
print(outputs_val)
print(states)


outputs_val:
[[[-0.5942823   0.37892666 -0.2408819   0.23835924  0.38741672]
  [ 0.4836566  -0.54892296 -0.77630025 -0.8152201   0.99998325]]

 [[-0.5457948   0.21758924 -0.46020284 -0.06179863  0.98796433]
  [-0.4132065   0.22017609 -0.41690663 -0.29208988 -0.40119433]]

 [[-0.49335387  0.04341494 -0.63480717 -0.35117978  0.99983394]
  [ 0.4635155  -0.56235206 -0.3522298  -0.80543756  0.99820757]]

 [[ 0.07905996 -0.99993193  0.2978742   0.37098378  0.99998844]
  [ 0.36765411 -0.54689246 -0.16637726 -0.9118903   0.6025403 ]]]
  
  
states: # 每一个batch的最后一个输出，即y1
[[ 0.4836566  -0.54892296 -0.77630025 -0.8152201   0.99998325]
 [-0.4132065   0.22017609 -0.41690663 -0.29208988 -0.40119433]
 [ 0.4635155  -0.56235206 -0.3522298  -0.80543756  0.99820757]
 [ 0.36765411 -0.54689246 -0.16637726 -0.9118903   0.6025403 ]]
```

However, this approach still builds a graph containing one cell per time step. If there were 50 time steps, the graph would look pretty ugly. It is a bit like writing a program without ever using loops, e.g.
```
(Y0=f(0,X0); Y1=f(Y0, X1); Y2=f(Y1, X2); ...; Y50=f(Y49, X50))
```

With such as large graph, you may even get **out-of-memory (OOM)** errors during backpropagation (especially with the limited memory of GPU cards), since it must store all tensor values during the forward pass so it can use them to compute gradients during the reverse pass.

# 3. Dynamic Unrolling Through Time
### no need to stack, unstack, or transpose

The ```dynamic_rnn()``` function uses a ```while_loop()``` operation to run over the cell the appropriate number of times, and you can set swap_memory=True if you want it to swap the GPU’s memory to the CPU’s
memory during backpropagation to avoid OOM errors. 

Conveniently, it also accepts a single tensor for all inputs at every time step ```(shape [None, n_steps, n_inputs])``` and it outputs a single tensor for all outputs at every time step``` (shape [None, n_steps, n_neurons])```; 
there is **no need to stack, unstack, or transpose**. 

The following code creates the same RNN as earlier using the ```dynamic_rnn()``` function. It’s so much nicer!

```python
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=x, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_val = sess.run(outputs, feed_dict={x:x_batch})
```
```python
print(outputs_val)



[[[ 8.3456165e-01  4.4506708e-01 -2.1010002e-01 -3.8176674e-01
    9.5100176e-01]
  [ 6.9185251e-01  9.9999988e-01  3.9719999e-01  2.0911363e-01
    9.9997133e-01]]

 [[ 9.7582531e-01  9.9783802e-01 -4.1931435e-01 -5.0139934e-01
    9.9969459e-01]
  [-8.3180672e-01 -2.0288916e-01  6.9902909e-01  4.1513512e-01
   -5.9112889e-04]]

 [[ 9.9668550e-01  9.9999386e-01 -5.9180784e-01 -6.0450792e-01
    9.9999809e-01]
  [-4.4303608e-01  9.9992198e-01  5.1096505e-01  5.6005251e-01
    9.9670297e-01]]

 [[-9.9883330e-01  9.9997902e-01  9.6597314e-01  1.2116167e-02
   -9.9381185e-01]
  [ 3.2501018e-01  9.8728287e-01 -7.4440765e-01  2.7665448e-01
    9.3733406e-01]]]
```

# 4. LSTM Cell

```python
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
```
```python
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)
```










