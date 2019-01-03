# numpy_kafka_sentence_generate_LSTM
Reference: [Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html)

## 1. Import and read data 
```python

import numpy as np
import matplotlib.pyplot as plt
from IPython import display  ## 动态显示loss plot
plt.style.use('seaborn-white')

data = open('../input/kafka.txt', 'r').read()

chars = list(set(data))
data_size = len(data)
X_size = len(chars)
print("data has %d characters, %d unique" % (data_size, X_size))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}
```
重点：```X_size = len(chars)```, 输入进LSTM单元的每个字母/汉字的one-hot向量长度/维度

## 2. Hyperparameters
```python
H_size = 100 # Size of the hidden layer
T_steps = 25 # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1 # Learning rate
weight_sd = 0.1 # Standard deviation of weights for initialization
z_size = H_size + X_size # Size of concatenate(H, X) vector
```
重点： 
1. ```H_size```：LSTM单元中的memory cell的向量长度/维度
2. ```T_step```：输入样本sample的长度，即一个输入样本sample/sentence有几个字母/汉字组成
3. ```z_size```：前一个LSTM的输出向量, 当前LSTM的输入（1个字母/汉字）one-hot向量，横向拼接 

## 3. Activation Functions and Derivatives
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def dsigmoid(y):
    return y * (1 - y)
    
def tanh(x):
    return np.tanh(x)
    
def dtanh(y):
    return 1 - y * y
```
## 4. Parameters
```python
class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value #parameter value
        self.d = np.zeros_like(value) #derivative
        self.m = np.zeros_like(value) #momentum for AdaGrad
```
We use random weights with normal distribution (`0`, `weight_sd`) for tanh activation function and (`0.5`, `weight_sd`) for sigmoid activation function.

Biases are initialized to zeros.

```python
class Parameters:
    def __init__(self):
        self.W_f = Param('W_f', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_f = Param('b_f', np.zeros((H_size, 1)))

        self.W_i = Param('W_i', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_i = Param('b_i', np.zeros((H_size, 1)))

        self.W_C = Param('W_C', np.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param('b_C', np.zeros((H_size, 1)))

        self.W_o = Param('W_o', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_o = Param('b_o', np.zeros((H_size, 1)))

        #For final layer to predict the next character
        self.W_v = Param('W_v', np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param('b_v', np.zeros((X_size, 1)))
        
    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
               self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]
        
parameters = Parameters()
```
## 5. Forward pass

![LSTM](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMvanillaFPcg.png)

```python
def forward(x, h_prev, C_prev, p = parameters):
    assert x.shape == (X_size, 1)
    assert h_prev.shape == (H_size, 1)
    assert C_prev.shape == (H_size, 1)
    
    z = np.row_stack((h_prev, x))
    f = sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
    i = sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
    C_bar = tanh(np.dot(p.W_C.v, z) + p.b_C.v)

    C = f * C_prev + i * C_bar  
    o = sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
    h = o * tanh(C)

    v = np.dot(p.W_v.v, h) + p.b_v.v
    y = np.exp(v) / np.sum(np.exp(v)) #softmax

    return z, f, i, C_bar, C, o, h, v, y
```
重点：
1. Parameter```x```: 输入样本sample/sentence中的一个字母/汉字的one-hot向量
2. Parameter```h_prev```: 前一个LSTM的输出（输出门乘以memory）, init 0 = ```h_prev = np.zeros((H_size, 1))```
3. Parameter```C_prev```: 前一个LSTM的memory（遗忘门乘以前一个memory+输入门乘以当前输入）,init 0 = ```C_prev = np.zeros((H_size, 1))```
4. Parameter```p```: 所有参数矩阵weights和偏置bias
5. ```z = np.row_stack((h_prev, x))```: 将前一个LSTM单元的输出，横向并上当前的输入（一个字母/汉字的one-hot向量） 
6. ```C = f * C_prev + i * C_bar```: 向量对应位置元素相加element-wise, how much to forget previous memory and how much to add current input to make new memory

## 6. Backward pass
![LSTM](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMvanillaBPcg2.png)
<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMvanillaBPformula.png">
</p>

```python
def backward(target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y, p = parameters):
    
    assert z.shape == (X_size + H_size, 1)
    assert v.shape == (X_size, 1)
    assert y.shape == (X_size, 1)
    
    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (H_size, 1)
        
    dv = np.copy(y)
    dv[target] -= 1

    p.W_v.d += np.dot(dv, h.T)
    p.b_v.d += dv

    dh = np.dot(p.W_v.v.T, dv)        
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    p.W_o.d += np.dot(do, z.T)
    p.b_o.d += do

    dC = np.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dtanh(C_bar) * dC_bar
    p.W_C.d += np.dot(dC_bar, z.T)
    p.b_C.d += dC_bar

    di = dC * C_bar
    di = dsigmoid(i) * di
    p.W_i.d += np.dot(di, z.T)
    p.b_i.d += di

    df = dC * C_prev
    df = dsigmoid(f) * df
    p.W_f.d += np.dot(df, z.T)
    p.b_f.d += df

    dz = (np.dot(p.W_f.v.T, df) + np.dot(p.W_i.v.T, di)
         + np.dot(p.W_C.v.T, dC_bar) + np.dot(p.W_o.v.T, do))
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC
    
    return dh_prev, dC_prev
```
重点：
1. Parameter ```dh_next```: dh'_[t], 下一个/后面一个LSTM往回传的 输出h 的导数， init = 0
2. Parameter ```dC_next```: dC'_[t], 下一个/后面一个LSTM往回传的 memory 的导数， init  0
3. Parameter ```C_prev```: C_[t-1], 下一个/后面一个LSTM往回传的 memory， init = 0
4. return ```dh_prev = dz[:H_size, :]```: dh'_[t-1], 再给 下一个/前面一个LSTM单元 回传的 输出h 的导数，shape = (0 ~ H_size行，所有列)
5. return ```dC_prev = f * dC```: dC'_[t-1], 再给 下一个/前面一个LSTM单元 回传的 memory 的导数

## 7. Loss & Forward-Backward pass

1. Clear gradients before each backward pass
```python
def clear_gradients(params = parameters):
    for p in params.all():
        p.d.fill(0)
```
2. Clip gradients to mitigate exploding gradients
```python
def clip_gradients(params = parameters):
    for p in params.all():
        np.clip(p.d, -1, 1, out=p.d)
```
3. Calculate and store the values in forward pass. Accumulate gradients in backward pass and clip gradients to avoid exploding gradients.

    * `input`, `target` are list of integers, with character indexes.
    * `h_prev` is the array of initial `h` at $h_{-1}$ (size H x 1)
    * `C_prev` is the array of initial `C` at $C_{-1}$ (size H x 1)
    * *Returns* loss, final $h_T$ and $C_T$

```python
def forward_backward(inputs, targets, h_prev, C_prev):
    global paramters
    
    # To store the values for each time step
    x_s, z_s, f_s, i_s,  = {}, {}, {}, {}
    C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
    v_s, y_s =  {}, {}
    
    # Values at t - 1 store the previous LSTM unit's output(h) and memory(C) into dict{},
    # which has its key=-1, in the forward pass process
    h_s[-1] = np.copy(h_prev)
    C_s[-1] = np.copy(C_prev)
    
    loss = 0
    # Loop through time steps
    assert len(inputs) == T_steps
    for t in range(len(inputs)):
        x_s[t] = np.zeros((X_size, 1))
        x_s[t][inputs[t]] = 1 # Input character
        
        (z_s[t], f_s[t], i_s[t],
        C_bar_s[t], C_s[t], o_s[t], h_s[t],
        v_s[t], y_s[t]) = \
            forward(x_s[t], h_s[t - 1], C_s[t - 1]) # Forward pass
            
        loss += -np.log(y_s[t][targets[t], 0]) # Loss for at t
        
    clear_gradients()

    dh_next = np.zeros_like(h_s[0]) #dh from the next character
    dC_next = np.zeros_like(C_s[0]) #dh from the next character

    for t in reversed(range(len(inputs))):
        # Backward pass
        dh_next, dC_next = \
            backward(target = targets[t], dh_next = dh_next,
                     dC_next = dC_next, C_prev = C_s[t-1],
                     z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                     C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
                     y = y_s[t])

    clip_gradients()
        
    return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]
```

BP could do withuot loss value??????????? YES, the design of cross entropy loss function makes its BP need no loss value.

重点：
1. Parameter ```inputs```: 输入的sample/sentence, time sequence, 有多少个字母/汉字组成的句子
2. Parameter ```h_prev```: 上一个/前面一个LSTM前向传播的 输出h， init = 0
3. Parameter ```C_prev```: 上一个/前面一个LSTM前向传播的 memory， init  0
4. ```for t in range(len(inputs)):```: 将每一个字母/汉字进行one-hot编码后，传入一个LSTM单元进行计算.inputs有几个字母/汉字组成，则循环调用几个LSTM单元进行计算。即，每一个字母/汉字需要一个LSTM单元。 每一个字母/汉字前向传播运算后，计算一次loss
5. 每一步（每一个字母/单词，每个LSTM单元）正向传播的门参数/中间变量计算后，存入一个dict{} 用于反向传播使用。

    ```{0: [ , , ...], 1: [ , , ...], ..., -1: h_prev}```, dict length = len(inputs) = T_steps = 25, 
    
    each key of the dict is the idx of each time step, 每个字母/汉字的idx，以及-1对应的初始值
    
    each value of the dict is a vector and has a length of (z_size, 1) = (H_size+X_size, 1)
6. ```dh_next```, ```dC_next```: 对于最后一个LSTM单元， 首次反向传播运算时（sample/sentence最后一个字母/汉字），其需要接收后面一个传回来的导数为0
7. return ```loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]```: 返回下一个sample/sentence输入时的loss 以及h_prev(h_[t-1])和C_prev(C_[t-1])

## 8. Sample the next character

```python
def sample(h_prev, C_prev, first_char_idx, sentence_length):
    x = np.zeros((X_size, 1))
    x[first_char_idx] = 1

    h = h_prev
    C = C_prev

    indexes = []
    
    for t in range(sentence_length):
        _, _, _, _, C, _, h, _, p = forward(x, h, C)
        idx = np.random.choice(range(X_size), p=p.ravel())
        x = np.zeros((X_size, 1))
        x[idx] = 1
        indexes.append(idx)

    return indexes
```
重点：
1. 随机选择一个corpus中的字母/单词，走一次正向传播，再从输出中随机挑选一个作为下一次的输入
2. ```for t in range(sentence_length):```: 根据输入参数```sentence_length```循环生成sentence_length（200）个idx
3. return ```indexes```: 用于下一次调用此函数时的输入参数```first_char_idx```

## 9. Training (Adagrad)
1. Update the graph and display a sample output **Loss plot**
```python
def update_status(inputs, h_prev, C_prev):
    #initialized later 仅声明变量，并未赋初始值，后面进行赋值
    global plot_iter, plot_loss
    global smooth_loss
    
    # Get predictions for 200 letters with current model
    sample_idx = sample(h_prev, C_prev, inputs[0], 200) # 根据初始idx，运行前向传播，指导生成200个idx
    txt = ''.join(idx_to_char[idx] for idx in sample_idx) # 将200个idx显示成字母/汉字

    # Clear and plot
    plt.plot(plot_iter, plot_loss)
    display.clear_output(wait=True)
    plt.show()

    #Print prediction and loss
    print("----\n %s \n----" % (txt, ))
    print("iter %d, loss %f" % (iteration, smooth_loss))
```

2. Update parameters
```python
def update_paramters(params = parameters):
    for p in params.all():
        p.m += p.d * p.d # Calculate sum of gradients
        #print(learning_rate * dparam)
        p.v += -(learning_rate * p.d / np.sqrt(p.m + 1e-8))Update parameters
```
重点：adagrad梯度下降, 优化参数，更新后的参数数值存在parameter.value里面

3. To delay the keyboard interrupt to prevent the training from stopping in the middle of an iteration 
```python
import signal

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
```

4. Main training loop

```python
# Exponential average of loss
# Initialize to a error of a random model
smooth_loss = -np.log(1.0 / X_size) * T_steps

iteration, pointer = 0, 0

# For the graph 为上面声明的gloabl变量进行初始赋值
plot_iter = np.zeros((0))
plot_loss = np.zeros((0))

while True:
    try:
        with DelayedKeyboardInterrupt():
            # Reset 当整个corpus走完一遍或者第一个iteration，将前一个LSTM的输出h和memory重置为0向量
            if pointer + T_steps >= len(data) or iteration == 0:
                g_h_prev = np.zeros((H_size, 1))
                g_C_prev = np.zeros((H_size, 1))
                pointer = 0
                
            # 输入句子，长度为T_steps=25, 25个字母/汉字一个句子
            # 每个输入字母/汉字的下一个字母/汉字作为标签
            inputs = ([char_to_idx[ch] for ch in data[pointer: pointer + T_steps]]) 
            targets = ([char_to_idx[ch] for ch in data[pointer + 1: pointer + T_steps + 1]]) 
            
            # 使用上面25个字母/汉字的句子进行训练，进行loss平滑输出
            # forward_backward()函数会对输入句子的每个字母/汉字计算loss并求和
            # 因为是基于时间time step的反向传播，每个时间就是一个字母/单词，所以对每个字母/单词都进行一次反向传播并计算loss
            loss, g_h_prev, g_C_prev = \
                forward_backward(inputs, targets, g_h_prev, g_C_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Print every hundred steps, 每训练一个25个字母/汉字的句子为一个iteration
            if iteration % 100 == 0:
                update_status(inputs, g_h_prev, g_C_prev)
            
            # adagrad梯度下降更新参数
            update_paramters()
            
            # 往plot_iter(nparray)中append元素， 类比list append
            plot_iter = np.append(plot_iter, [iteration]) # np array中存放每个iteration的数值0,1,2，...
            plot_loss = np.append(plot_loss, [loss]) # np array中存放每个iteration的loss

            pointer += T_steps # 指针增加一个sentence的长度
            iteration += 1
    except KeyboardInterrupt:
        update_status(inputs, g_h_prev, g_C_prev)
        break
```
重点： 未指定训练多少次可以走完一个完整的Corpus/数据集，使用while True循环，当感觉到loss达到可接受程度，手动终止训练

> 走完1次全部corpus是一个iteration，每训练完100个iteration/走完100次全部数据集, 使用训练出来的参数生成一个200字母/汉字的句子并打印，不是100取整的iteration时，不执行打印，进行下面步骤


## 10. Gradient Check

Approximate the numerical gradients by changing parameters and running the model. Check if the approximated gradients are equal to the computed analytical gradients (by backpropagation).

Try this on `num_checks` individual paramters picked randomly for each weight matrix and bias vector.

1. Calculate numerical gradient
```python
from random import uniform

def calc_numerical_gradient(param, idx, delta, inputs, target, h_prev, C_prev):
    old_val = param.v.flat[idx]
    
    # evaluate loss at [x + delta] and [x - delta]
    param.v.flat[idx] = old_val + delta
    loss_plus_delta, _, _ = forward_backward(inputs, targets,
                                             h_prev, C_prev)
    param.v.flat[idx] = old_val - delta
    loss_mins_delta, _, _ = forward_backward(inputs, targets, 
                                             h_prev, C_prev)
    
    param.v.flat[idx] = old_val #reset

    grad_numerical = (loss_plus_delta - loss_mins_delta) / (2 * delta)
    # Clip numerical error because analytical gradient is clipped
    [grad_numerical] = np.clip([grad_numerical], -1, 1) 
    
    return grad_numerical
```

2. Check gradient of each paramter matrix/vector at `num_checks` individual values
```python
def gradient_check(num_checks, delta, inputs, target, h_prev, C_prev):
    global parameters
    
    # To calculate computed gradients
    _, _, _ =  forward_backward(inputs, targets, h_prev, C_prev)
    
    
    for param in parameters.all():
        #Make a copy because this will get modified
        d_copy = np.copy(param.d)

        # Test num_checks times
        for i in range(num_checks):
            # Pick a random index
            rnd_idx = int(uniform(0, param.v.size))
            
            grad_numerical = calc_numerical_gradient(param,
                                                     rnd_idx,
                                                     delta,
                                                     inputs,
                                                     target,
                                                     h_prev, C_prev)
            grad_analytical = d_copy.flat[rnd_idx]

            err_sum = abs(grad_numerical + grad_analytical) + 1e-09
            rel_error = abs(grad_analytical - grad_numerical) / err_sum
            
            # If relative error is greater than 1e-06
            if rel_error > 1e-06:
                print('%s (%e, %e) => %e'
                      % (param.name, grad_numerical, grad_analytical, rel_error))
```
```python
gradient_check(10, 1e-5, inputs, targets, g_h_prev, g_C_prev)
```
