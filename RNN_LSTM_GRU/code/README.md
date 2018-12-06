
# numpy_kafka_sentence_generate_LSTM.ipynb
Reference: [Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html)

## 1. Import and read data
```python

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
plt.style.use('seaborn-white')

data = open('../input/kafka.txt', 'r').read()

chars = list(set(data))
data_size, X_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_size, X_size))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}
```
## 2. Hyperparameters
```python
H_size = 100 # Size of the hidden layer
T_steps = 25 # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1 # Learning rate
weight_sd = 0.1 # Standard deviation of weights for initialization
z_size = H_size + X_size # Size of concatenate(H, X) vector
```
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
## 6. Backward pass
![LSTM](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMvanillaBPcg.png)
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







# numpy_kafka_sentence_generate_RNN.ipynb

[kaggle kernel](https://www.kaggle.com/davidkor/numpy-kafka-sentence-generate-rnn)

## 1. Data Preparation
```python
data = open('../input/kafka.txt','r').read()
chars = sorted(list(set(data)))
data_size = len(data)
vocab_size = len(chars)
print('data_size: ',data_size, '\nvocab_size: ',vocab_size)

# data_size:  118561 
# vocab_size:  63
```

```python
char_to_idx = { char:idx for idx,char in enumerate(chars)}
idx_to_char = { idx:char for idx,char in enumerate(chars)}
print('char_to_idx: ',char_to_idx, '\nidx_to_char: ',idx_to_char)
```
## 2. Hyper Parameters

```python
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

Wxh = np.random.randn(hidden_size,vocab_size) * 0.01 # weight matrix input x->hidden
Whh = np.random.randn(hidden_size,hidden_size) * 0.01 # weight matrix hidden->hidden/memory
Why = np.random.randn(vocab_size,hidden_size) * 0.01 # weight matrix hidden->output y
bh = np.zeros((hidden_size,1)) # bias of hidden
by = np.zeros((vocab_size,1)) # bias of output y
```
## 3. Loss Function
```python
def lossFunc(inputs, targets, hprev):
    # inputs: (25, 63, 1), a sentence, contains 25 words, which word has a (vocab,1) shape vector
    # outputs: (25, 63, 1), the label of input which has the same shape of input
    # hprev: the previous state of hodden layer / memory
    xs, hs, ys, ps = {}, {}, {}, {} # state of x, hidden, y, p(probability of y)
    hs[-1] = np.copy(hprev) # init previous state of hidden/memory in dict {-1:hprev}
    loss = 0
    
    # Forward
    for t in range(len(inputs)): # idx of each word in input sentence / each time step
        xs[t] = np.zeros((vocab_size,1)) 
        xs[t][inputs[t]] = 1 # t-th word's vector's t-th element = 1 
        hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1]+bh))
        ys[t] = np.dot(Why, hs[t])+by
        ps[t] = np.exp(ys[t]) / (np.sum(np.exp(ys[t]))) # ps[t] 向量的每个元素对应词汇表中每个单词的概率
        loss += -np.log(ps[t][targets[t],0]) # ps[t][targets[t]]选择出label对应盖茨的概率, 0 ???
    
    # Backward
    # Below Conputaional Graph
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0]) # hs[3] also Ok, cuz each vector has same shape
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t]) # back softmax-crossentropy
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # if variable being inited above, use +=, else =
        dhraw = (1-hs[t]*hs[t]) * dh # (tanh)'=1 - (tanh)^2
        dbh += dhraw
        dWhh += np.dot(dhraw, hs[t-1].T)
        dWxh += np.dot(dhraw, xs[t].T)
        dhnext += np.dot(Whh.T, dhraw)
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # eliminate gradient vanishing, exploding
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1] 
    # hs[len(inputs)-1]: last second hidden state/memory of this input sentence
    # hs[len(inputs)-1]: the next hprev, when this lossFunc() is invoked recursively.
```

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBPcomputationalgraph.png)

## 4. Sampling - sentence generation

```python
def sample(h, seed_ix, n):
    # h: last hidden state / memory
    # seed_idx: the idx of the first word/char of the sentence we want to generate in corpus
    # n: the length of the sentence we want to generate, how many characters to predict
    
    # create the first word's/char's vector
    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = [] # resotre the idx of words/chars of the sentence
    
    for t in range(n):
        h = np.tanh(np.dot(Wxh,x) + (np.dot(Whh, h)+bh))
        y = np.dot(Why, h)+by
        p = np.exp(y) / np.sum(np.exp(y))
        # select the biggest element? NO!NO! select randomly
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    txt = ''.join(idx_to_char[ix] for ix in ixes)
    print('-----\n',txt,'\n-----')
    
hprev = np.zeros((hidden_size,1))
sample(hprev,char_to_idx['a'],100)
"""
-----
 Lei'ma;FQkL:Fl O;CI?MHexkjEkr
THfa!;'NfLyyC"nPdp?QIBLgu(w LUtjNyeHix?)qGA?)ym;fTVqC
OkD?;rUlooPmCvCT 
-----
"""
```
## 5. Model Training
```python
n = 0 # iteration
p = 0 # idx/position of 1st word
# memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0                

while n <= 1000 * 100:
    if p+1+seq_length >= len(data) or n ==0:
        hprev = np.zeros((hidden_size,1))
        p = 0
        
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

    # forward seq_length characters through the net and fetch gradient                                                                                                                          
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunc(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # sample from the model now and then                                                                                                                                                        
    if n % 1000 == 0:
        print('iter: {} - loss: {}'.format(n, smooth_loss)) # print progress
        sample(hprev, inputs[0], 200)

    # perform parameter update with Adagrad                                                                                                                                                     
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update       
        # new param(Wxh, Whh, Why, bh, by) will be used in lossFunc() and sample()
        
    p += seq_length
    n += 1
```
