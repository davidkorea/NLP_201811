# word2vec

based on **distributed hypothesis**，分布式假设。即，一个单词的和其周围的单词语义更相近。

# 1. Skip-gram
我的理解是，类似于ngram，两个单词总是同时出现，因此，我就使得，给定中心词，其周边词出现的概率P（周围词|中心词）越大越好。而在求得这个概率最大的同时，也训练出了weights，拿此weights当做词向量

中心词，预测周围的词
