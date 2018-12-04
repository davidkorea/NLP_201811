
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
    #  hs[len(inputs)-1]: last second hidden state/memory of this input sentence
```

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBPcomputationalgraph.png)
