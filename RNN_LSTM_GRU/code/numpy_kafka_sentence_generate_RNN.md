# numpy_kafka_sentence_generate_RNN

[kaggle kernel](https://www.kaggle.com/davidkor/numpy-kafka-sentence-generate-rnn) <- need to be fixed  

## 1. Data Preparation
```python
data = open('../input/kafka.txt','r').read()
chars = sorted(list(set(data))) 
# unlike the LSTM above, 经过排序后的corpus每次读取data后的单词顺序fixed，即idx不会发生变化
data_size = len(data) # all text with duplicated chars
vocab_size = len(chars) # unduplicated chars
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
    # inputs: 由25个单词的idx组成的list
    # ---X---inputs: (25, 63, 1), a sentence, contains 25 words, which word has a (vocab,1) shape vector---X---
    # outputs: (25, 63, 1), the label of input which has the same shape of input
    # hprev: np.zeros((hidden_size,1))，the previous state of hodden layer / memory
    # 此处inputs，xs，hs，ys，ps的‘s’表示为复数的含义，即一组输入，一个输入句子包含多个单词
    xs, hs, ys, ps = {}, {}, {}, {} # state of x, hidden, y, p(probability of y)
    hs[-1] = np.copy(hprev) # init previous state of hidden/memory in dict {-1:hprev}
    loss = 0
    
    # Forward
    for t in range(len(inputs)): # idx of each word in inputs sentence / each time step
        xs[t] = np.zeros((vocab_size,1)) # 定义一个vector
        xs[t][inputs[t]] = 1 # t-th word's vector's t-th element = 1，
                             # xs[t]表示在dict中取出一个vector（上一行已定义），
                             # xs[t][inputs[t]] = 1为一个vector中的某一位，置为1.
        hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1]) + bh ) 
        # 当前输入x[t]放射变换 element-wise add对应位置元素相加 前一个隐藏层h[t-1]放射变换 
        # 加和后的向量经过tanh非线性变化
        ys[t] = np.dot(Why, hs[t])+by
        ps[t] = np.exp(ys[t]) / (np.sum(np.exp(ys[t]))) # ps[t] 向量的每个元素对应词汇表中每个单词的概率
                                                        # 即，输入一个单词，预测下一个单词应该是词表vocab中哪个单词，
        loss += -np.log(ps[t][targets[t],0]) # ps[t][targets[t]]选择出label对应targets的概率, 0 ???
                                             # ps[t] = [ 
                                                         [0.003],  
                                                         [0.012],
                                                         ...
                                                         [0.045]
                                                        ]
                                             # 如果不加0，则只选中了一个list，并不是list里面的数值
    
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
        h = np.tanh(np.dot(Wxh,x) + np.dot(Whh, h) + bh ) # ---错误更正---
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
# hprev=0, x初始化为onehot, Wxh, Whh，bh最开始已经初始化randn和zero

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
        
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]] # 将输入句子中的每个单词转换为idx
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
        mem += dparam * dparam # 操作这一步只是为了套用adagrad方法，目的是用于更新下面的param，即[Wxh, Whh, Why, bh, by]
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update       
        # new param(Wxh, Whh, Why, bh, by) will be used in lossFunc() and sample()
        
    p += seq_length
    n += 1
```

