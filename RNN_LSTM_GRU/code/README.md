
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



![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBPcomputationalgraph.png)
