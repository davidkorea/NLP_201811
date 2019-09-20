# 1. Word Embedding
## 1.1 Count based
if two words co-occur frequently, the vector of the the words would be similar(inner product)

- Glove

## 1.2 predict based
if we have enough training data, skipgram maybe better than CBOW, because skipgram can generate more training sample pairs than CBOW.

https://blog.csdn.net/leadai/article/details/80249999

`the cat sat on the mat`, window_size = 2

- CBOW, **8 samples**
  - the: [(cat, sat), the]
  - cat: [(sat, on), cat]
  - sat: [(the, cat), sat], [(on, the), sat]
  - on: [(cat, sat), on]. [(the, mat), on]
  - the: [(sat, on), the]
  - mat: [(on, the), mat]

- Skipgram, **18 samples**
  - the: (the, cat), (the sat)
  - cat: (cat, the), (cat, sat), (cat, on)
  - sat: (sat, the), (sat, cat), (sat, on), (sat, the)
  - on: (on, cat), (on, sat), (on, the), (on, mat)
  - the: (the, sat), (the, on), (the, mat)
  - mat: (mat, on), (mat, the)


# word2vec
[Word2Vec word embedding tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/)



based on **distributed hypothesis**，分布式假设。即，一个单词的和其周围的单词语义更相近。

## 1. Skip-gram
我的理解是，类似于ngram，两个单词总是同时出现，因此，我就使得，给定中心词，其周边词出现的概率P（周围词|中心词）越大越好。而在求得这个概率最大的同时，也训练出了weights，拿此weights当做词向量

- 不具备上下文语义，所有语境下都是相同的向量
- 降维后，不能实现同一类名词，映射到相近的空间，apple banana

中心词，预测周围的词, `the cat sat on the mat`, windows_size=1

- the: (the, cat) 
- cat: (cat, the), (cat, sat)
- sat：(sat, cat), (sat, on)

```python
input = ['the', 'cat', 'cat', 'sat', 'sat']
label = ['cat', 'the', 'sat', 'cat', 'on']
```
