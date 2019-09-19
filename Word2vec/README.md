# 1. Word Embedding
## 1.1 Count based

## 1.2 predict based
- CBOW
- Skipgram



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
