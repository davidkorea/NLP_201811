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

![](https://i.loli.net/2019/01/11/5c3847adde410.png)


# 2. RNN Model
First, let’s create the RNN. It will contain 100 recurrent neurons and we will unroll it over 20 time steps
since each training instance will be 20 inputs long. Each input will contain only one feature (the value at
that time). The targets are also sequences of 20 inputs, each containing a single value. The code is almost
the same as earlier:

## 2.1 Using an OuputProjectionWrapper
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
> **NOTE**:
> In general you would have more than just one input feature. For example, if you were trying to predict stock prices, you would likely have many other input features at each time step, such as prices of competing stocks, ratings from analysts, or any other feature that might help the system make its predictions.

At each time step we now have an output vector of size 100. But what we actually want is a single output value at each time step. The simplest solution is to wrap the cell in an ```OutputProjectionWrapper```. 

A cell wrapper acts like a normal cell, proxying every method call to an underlying cell, but it also add ssome functionality. The ```OutputProjectionWrapper``` adds a fully connected layer of linear neurons (i.e., without any activation function) on top of each output (but it does not affect the cell state). All these fully connected layers share the same (trainable) weights and bias terms. 

![](https://i.loli.net/2019/01/11/5c38483442d82.png)

Wrapping a cell is quite easy. Let’s tweak the preceding code by wrapping the BasicRNNCell into an ```OutputProjectionWrapper```:
```python
cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
            output_size=n_outputs)
```
So far, so good. Now we need to define the cost function. We will use the Mean Squared Error (MSE), as we did in previous regression tasks. Next we will create an Adam optimizer, the training op, and the variable initialization op, as usual:
```python
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
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
with tf.Session() as sess:                          # not shown in the book
    saver.restore(sess, "./my_time_series_model")   # not shown

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
```
```python
plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "y*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()
```
![](https://i.loli.net/2019/01/11/5c3848af3ddb3.png)

## 2.2 More efficient without OutputProjectionWrappe

Although using an OutputProjectionWrapper is the simplest solution to reduce the dimensionality of the RNN’s output sequences down to just one value per time step (per instance), it is not the most efficient. There is a trickier but more efficient solution: you can reshape the RNN outputs from ```[batch_size, n_steps, n_neurons]``` to ```[batch_size * n_steps, n_neurons]```, then apply a single fully connected layer with the appropriate output size (in our case just 1), which will result in an output tensor of shape ```[batch_size * n_steps, n_outputs]```, and then reshape this tensor to ```[batch_size, n_steps, n_outputs]```. 

To implement this solution, we first revert to a basic cell, without the OutputProjectionWrapper:
```python
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```
Then we stack all the outputs using the reshape() operation, apply the fully connected linear layer
(without using any activation function; this is just a projection), and finally unstack all the outputs, again
using reshape():
```python
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs,
                                  activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
```
The rest of the code is the same as earlier. This can provide a significant speed boost since there is just
one fully connected layer instead of one per time step.

![](https://i.loli.net/2019/01/11/5c384aec71abb.png)

# 3. Creative RNN
Now that we have a model that can predict the future, we can use it to generate some creative sequences,
as explained at the beginning of the chapter. All we need is to provide it a seed sequence containing
n_steps values (e.g., full of zeros), use the model to predict the next value, append this predicted value
to the sequence, feed the last n_steps values to the model to predict the next value, and so on. This
process generates a new sequence that has some resemblance to the original time series.

```python
with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "r-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "r-", linewidth=3)
plt.xlabel("Time")

plt.show()
```
![](https://i.loli.net/2019/01/11/5c38563364706.png)
