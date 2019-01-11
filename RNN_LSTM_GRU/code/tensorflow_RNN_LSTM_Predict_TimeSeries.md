# tensorflow_RNN_LSTM_Predict_TimeSeries

Now let’s take a look at how to handle time series, such as stock prices, air temperature, brain wave patterns, and so on. In this section we will train an RNN to predict the next value in a generated time series. Each training instance is a randomly selected sequence of 20 consecutive values from the time series, and the target sequence is the same as the input sequence, except it is shifted by one time step into the future.

# 1. Time series

```python
t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    # 随即生成size为（batch_size, 1）的array 
    
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    # np.arange(0., n_steps + 1) 从0到n_steps=20每间隔1生成的整数array（n_steps+1，1）
    # t0（batch_size, 1）+（n_steps+1，1）=（batch_size，n_steps+1）
    
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
    # 0~倒数第二个元素为x，1~最后一个为target
```
**重点**

1. numpy加法：（batch_size, 1）+（n_steps+1，1）=（batch_size，n_steps+1）

```python
# 此处并没有用到上面的next_batch()

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution)) # 定义域t（0，30）共30/0.1=300个离散点

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
# 在上面定义域t的范围中，选取长度为n_steps=20的一段最为示例

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t * \sin(t) / 3 + 2 * \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "r-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122) # 画出示例的x 和 target
plt.title("A training instance", fontsize=14)
# t_instance[:-1] 0~倒数第二个
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# t_instance[1:] 1~最后一个
plt.plot(t_instance[1:], time_series(t_instance[1:]), "y*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")
# save_fig("time_series_plot")
plt.show()
```

![](https://www.kaggleusercontent.com/kf/9404547/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..oBX1fUW5X1zwDzYFkwYDUA.OrB5t7zKCfaiYqvmKG0Iucf8mnydrfhrtao76sFIqyDcxDjOXn-jj1og_h5YtosCnKFh2cGopr81ZRRK2SAvpRdFdxVMoKHYwOI9WPRDJRK927r82AxMnaUeTEXFmfR9nellwGI6kek4nonzJlE54fIRMDuOp-Moq8gqiAZ9_og.Usf_cMPSK_J_1yS1rgNccg/__results___files/__results___13_0.png)


# 2. RNN
First, let’s create the RNN. It will contain 100 recurrent neurons and we will unroll it over 20 time steps
since each training instance will be 20 inputs long. Each input will contain only one feature (the value at
that time). The targets are also sequences of 20 inputs, each containing a single value. The code is almost
the same as earlier:

## 2.1 No output wrapper
```python
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```
> **NOTE**:
> In general you would have more than just one input feature. For example, if you were trying to predict stock prices, you would likely have many other input features at each time step, such as prices of competing stocks, ratings from analysts, or any other feature that might help the system make its predictions.

At each time step we now have an output vector of size 100. But what we actually want is a single output value at each time step. The simplest solution is to wrap the cell in an ```OutputProjectionWrapper```. 

## 2.2 Using an OuputProjectionWrapper

A cell wrapper acts like a normal cell, proxying every method call to an underlying cell, but it also add ssome functionality. The ```OutputProjectionWrapper``` adds a fully connected layer of linear neurons (i.e., without any activation function) on top of each output (but it does not affect the cell state). All these fully connected layers share the same (trainable) weights and bias terms. 

Wrapping a cell is quite easy. Let’s tweak the preceding code by wrapping the BasicRNNCell into an ```OutputProjectionWrapper```:
```python
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
# cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
            output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```
So far, so good. Now we need to define the cost function. We will use the Mean Squared Error (MSE), as we did in previous regression tasks. Next we will create an Adam optimizer, the training op, and the variable initialization op, as usual:
```python
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
```
Now on to the execution phase:
```python
saver = tf.train.Saver()

n_iterations = 200000
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(iteration, "\tMSE:", mse)
    
    saver.save(sess, "./my_time_series_model") # not shown in the book
```
The program’s output should look like this:
```
199500 	MSE: 0.07861377
199600 	MSE: 0.079538494
199700 	MSE: 0.09140773
199800 	MSE: 0.08307927
199900 	MSE: 0.09161533
```
Once the model is trained, you can make predictions:
```python

```








