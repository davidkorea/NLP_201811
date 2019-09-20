# 1. Word Embedding

**You shall know a word by the company it keeps.** 阅读理解，根据上下文推测一个单词含义

## 1.1 Count based
if two words co-occur frequently, the vector of the the words would be similar(inner product)

- Glove

## 1.2 predictiion based

输入是一个单词的one-hot，one-hot的每一维都是一个输入，和权重相乘后，作为隐藏层的输入，输出是词库中每个单词的下一个单词的概率。取第一个隐藏层的输入向量z为该单词的词向量。

<img src="http://wx3.sinaimg.cn/large/006gDTsUgy1g75tl0ur6zj30hc0cv79k.jpg" width="500" data-width="500" >

示例中，无论输入是马英九还是蔡英文，输出都是宣誓就职。不同输入，如何才能使输出中“宣誓就职”的概率都最大呢？答案就是，经过weight变换后的向量应该在相近的空间，才能使得输入同样取得的最大概率。而相近空间的z作为词向量，就可以表达相近词义的单词的向量也比较接近。

<img src="http://wx1.sinaimg.cn/large/006gDTsUgy1g75tnpfer9j30hc0ctgrn.jpg" width="500" data-width="500" >

而只看一个单词，其下一个单词的可能性是千千万万的。那么使用前2个，前3个单词预测下一个单词。但是每个单词的权重矩阵需要相同，及w_t-1的第一维和隐藏层第一个neuron的权重 与 w_t-2的第一维和隐藏层第一个neuron的权重 需要一直。
- 其中一个原因是，交换w_t-1和w_t-2两个单词的位置，可以获得同样的词向量z，但如果权重矩阵不一样，那么交换位置后的词向量也将变化。
- 另一个原因是，每次输入一般都是十万维，因此参数超级多，使用同样权重矩阵可以减少训练参数

<img src="http://ws4.sinaimg.cn/large/006gDTsUgy1g75u5n08nkj30gt0bmwib.jpg" width="500" data-width="500" >

初始化两个权限矩阵需要完全相同，w_i 和 w_j 的求导也需要完全相同，即更新参数时需要同时减去两个参数，以保证更新后的w_i 和 w_j 也相同

<img src="http://ws3.sinaimg.cn/large/006gDTsUgy1g75x9f47ibj30z20cngv1.jpg" width="1262" data-width="1262" >

使用前面2个单词预测后面的单词，但是此处还不是CBOW，因为CBOW使用 前一个 和 后一个 来预测 中心词
<img src="http://ws1.sinaimg.cn/large/006gDTsUgy1g75xfrkyaej30hl093tc8.jpg"  width="500" data-width="500">

predictiion based 的其它形式
- CBOW
- Skipgram


# word2vec
[Word2Vec word embedding tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/)


if we have enough training data, skipgram maybe better than CBOW, because skipgram can generate more training sample pairs than CBOW.

https://blog.csdn.net/leadai/article/details/80249999

### CBOW, **8 samples**
  - the: [(cat, sat), the]
  - cat: [(sat, on), cat]
  - sat: [(the, cat), sat], [(on, the), sat]
  - on: [(cat, sat), on]. [(the, mat), on]
  - the: [(sat, on), the]
  - mat: [(on, the), mat]
  
### Skipgram, **18 samples**
  - the: (the, cat), (the sat)
  - cat: (cat, the), (cat, sat), (cat, on)
  - sat: (sat, the), (sat, cat), (sat, on), (sat, the)
  - on: (on, cat), (on, sat), (on, the), (on, mat)
  - the: (the, sat), (the, on), (the, mat)
  - mat: (mat, on), (mat, the)
  
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
