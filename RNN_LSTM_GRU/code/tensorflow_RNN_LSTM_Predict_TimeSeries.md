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
    # np.arange(0., n_steps + 1) 从0到n_steps=20每间隔1生成的整数array（n_steps，1）
    # t0（batch_size, 1）+（n_steps，1）=（batch_size，n_steps）
    
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
```
**重点**

1. numpy加法：（batch_size, 1）+（n_steps，1）=（batch_size，n_steps）

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
