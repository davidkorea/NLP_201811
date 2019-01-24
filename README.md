# NLP_201811
# 2019-01-24
## 1. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
![](https://i.loli.net/2019/01/24/5c49834daf3ab.png)

## 2. OFFICIAL: [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt#introduction)

![](https://i.loli.net/2019/01/24/5c49834d77c27.png)


## 3. [深度学习中的注意力机制](https://mp.weixin.qq.com/s?__biz=MzA4Mzc0NjkwNA==&mid=2650783542&idx=1&sn=3846652d54d48e315e31b59507e34e9e&chksm=87fad601b08d5f17f41b27bb21829ed2c2e511cf2049ba6f5c7244c6e4e1bd7144715faa8f67&mpshare=1&scene=1&srcid=1113JZIMxK3XhM9ViyBbYR76#rd)

![](http://mmbiz.qpic.cn/mmbiz_png/ptp8P184xjxeRHqppry03SX1TTiblocHfEic80ZyYfA1hF6F58uYTKHl7g8tn90MFIQZpNtCJHUjG1O9jYkwsnNA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



# 2019-01-23
1. [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - [CN](https://www.jianshu.com/p/e7d8caa13b21)
3. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

# 2019-01-22
## 1. BERT
1. [NLP seq2seq BERT blogs](https://www.jianshu.com/u/abfe703a00fe)
2. [github of the above blog author](https://github.com/FuYanzhe2)
2. [google-research/bert](https://github.com/google-research/bert)
## 2. [Tensorflow nmt源码解析](https://blog.csdn.net/stupid_3/article/details/78956470)
## 3. [运行tensorflow_NMT模型](https://blog.csdn.net/Moluth/article/details/79142689)
3.怎么用nmt
3.1 数据格式

仿照着之前下载的8个文件，做好数据对应，其中三个文件是英文的一句一行单词之间通过空格分隔，还有三个是越南语，格式和英语一样。vocab.vi 是越南语的词汇表取了常用的5000个词,vocab.en 是英语词汇表取了最常用的前5000个词语，但是它们前三个词语是<unk> 代表不认识的词语 <s>开始 </s>结束，这三个词必须在词汇表中否则nmt模型不能工作，具体原因官方github上有解释。

3.2 模型参数
```
python -m nmt.nmt --src=en --tgt=vi --vocab_prefix=/tmp/nmt_data/vocab  --train_prefix=/tmp/nmt_data/train --dev_prefix=/tmp/nmt_data/tst2012  --test_prefix=/tmp/nmt_data/tst2013 --out_dir=/tmp/nmt_model --num_train_steps=12000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=bleu
```
这条命令中只是使用了个别的参数，还有一些其他有用的参数，如下：
```
forget_bias=1.0 这个是lstm的记忆力参数,取值范围在[0.0,1.0]越大代表记性越好

batch_size=128 这个代表每次训练128条数据，取值是正整数，如果太大，需要的内存会增大

learning_rate=1 学习率，正常情况下设置成小于等于1的值，默认值 1

num_gpus=1 机器中gpu个数，默认值是1

eos='</s>' 结束符配置成</s>,参考3.1 数据格式

sos='<s>' 同上，这两个参数没有配置的必要

src_max_len=50 源输入最大长度，针对我们训练的英语-越南语模型中，意思是每行最长接受50个英语单词，其余忽略

tgt_max_len=50 目标输出最大长度，默认值50.这个和上面的参数有时很有用，假设我们要做文章摘要，参数可以这样写--src_max_len=800 --tgt_max_len=150，这两个参数都会影响训练和预测速度，他们越大，模型跑的越慢。

share_vocab=False 这个意思是是否公用词汇表，假设做文章摘要，把这个设置成True。因为不是做翻译，输入和输出是同一种语言。
```
还有一些其他参数，不再列举，可以去源代码中nmt.py文件中查看。


3.3 训练一个聊天机器人（汉语）
3.3.1 准备好训练数据，开发数据，测试数据，汉语常用汉字表（前5000个）即可

仿照3.1中的数据，来准备训练数据。这次不是翻译数据，而是对话数据。比如：

```
train.src

你 好 ！

很 高 兴 认 识 你 。

当 然 很 激 动 了。
```

```
train.tgt

你 好 呀 ！

我 也 是 呢 ， 你 有 没 有 很 激 动 。

激 动 你 妹 啊 。
```
```
vocab.src

<unk>
<s>
</s>
，
的
。
<sp>
一
0
是
1
、
在
有
不
了
2
人
中
大
国
年
```

3.3.2 接下来进行训练

```
python -m nmt.nmt --src=src --tgt=tgt --vocab_prefix=/tmp/chat_data/vocab  --train_prefix=/tmp/chat_data/train --dev_prefix=/tmp/chat_data/dev  --test_prefix=/tmp/chat_data/test --out_dir=/tmp/nmt_model --num_train_steps=192000 --steps_per_stats=100 --num_layers=2 --num_units=256 --dropout=0.2 --metrics=bleu --src_max_len=80 --tgt_max_len=80 --share_vocab=True
```
经过漫长的训练，聊天模型训练完毕

3.3.3 集成到项目

有三种方案将训练的模型集成到项目中：

（1）对nmt进行部分修改，在项目代码中调用预测，使结果以文件形式展示，然后去文件中提取结果。优点：改动少，可以快速集成。 缺点：运行速度很慢

（2）对nmt进行部分修改，在项目代码中调用预测，只是要给nmt的源代码添加参数和返回值，返回值就是结果。 优点：改动少，可以快速集成。缺点：运行速度慢

（3）把nmt重构，写成一个对象，不要释放session，这样调用的速度会快一些。优点：运行速度快。 缺点：需要对nmt进行深入了解，开发周期长

前两种速度慢的原因是，每次运行都要加载大量的参数，加载词汇。第一种方案还多进行了两次io操作。


# 2019-01-21

# 2019-01-20

# 2019-01-18

1. [Neural Machine Translation using word level language model and embeddings in Keras](https://github.com/devm2024/nmt_keras/blob/master/base.ipynb)
2. [simple sequence-to-sequence model with dynamic unrolling](https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb)

# 2019-01-17 seq2seq

1. [Sequence to Sequence (seq2seq) Recurrent Neural Network (RNN) for Time Series Prediction](https://github.com/guillaume-chevalier/seq2seq-signal-prediction)
2. OFFICIAL: [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt#introduction)

3. [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
4. [A simple Seq2Seq translator](https://www.kaggle.com/jannesklaas/a-simple-seq2seq-translator)

# 2019-01-15 Attention Mechanism
1. [深度学习中的注意力机制](https://mp.weixin.qq.com/s?__biz=MzA4Mzc0NjkwNA==&mid=2650783542&idx=1&sn=3846652d54d48e315e31b59507e34e9e&chksm=87fad601b08d5f17f41b27bb21829ed2c2e511cf2049ba6f5c7244c6e4e1bd7144715faa8f67&mpshare=1&scene=1&srcid=1113JZIMxK3XhM9ViyBbYR76#rd)
2. OFFICIAL: [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt#introduction)
3. [Google’s NMT (GNMT) system](https://ai.google/research/pubs/pub45610)
4. [Thang Luong's Thesis on Neural Machine Translation](https://github.com/lmthang/thesis)

# 2019-01-14 Neural Machine Translation
OFFICIAL: [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt#introduction)

# 2019-01-13
[tensorflow_Deep_RNN](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_Deep_RNN.md)


# 2019-01-12 tensorflow does NOT support python3.7

## 1. anaconda create a new env and install python3.6 
https://github.com/davidkorea/NLP_201811/issues/11
## 2. anaconda create python3.6 new environment and new kernel 
https://github.com/davidkorea/NLP_201811/issues/12
### 2.1. create a new environment and install tensorflow 

1. ```conda create -n tensorflow pip python=3.6``` create new env

2. ```conda activate tensorflow``` activate this new env

3. ```pip install --ignore-installed --upgrade tensorflow``` install tensorflow in this new env

### 2.2. create a kernel in this environment

1. ```source activate tensorflow``` activate new env

2. ```(tensorflow) DaviddeMacBook-Pro:~ david$ conda install ipykernel``` install ipykernel in this env

3. ```(tensorflow) DaviddeMacBook-Pro:~ david$ python -m ipykernel install --user --name tensorflow --display-name "tensorflow"```  create a new kernel in this env, Installed kernelspec tensorflow in /Users/david/Library/Jupyter/kernels/tensorflow

4. ```(tensorflow) DaviddeMacBook-Pro:~ david$ jupyter notebook``` open jpynb in this env


# 2019-01-11

1. Issue on mac install pip3 https://github.com/davidkorea/NLP_201811/issues/8
2. Issue on mac python3.7 install tensorflow https://github.com/davidkorea/NLP_201811/issues/9
3. Issue on mac anaconda python3.7 install tensorflow https://github.com/davidkorea/NLP_201811/issues/10

4. [tensorflow_RNN_LSTM_Predict_TimeSeries](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_RNN_LSTM_Predict_TimeSeries.md)

# 2019-01-10

https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2.2/full_code.py

# 2019-01-09
hands on examples codes: https://github.com/ageron/handson-ml

# 2019-01-08

# 2019-01-07


# 2019-01-06



# 2019-01-04



# 2019-01-03

1. [numpy_kafka_sentence_generate_LSTM](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_LSTM.md)

It is better to read the below RNN toturail first to get the main code structure and thoughts for knowing the basics, and then go to this toturial. tips fior this toturial is that go through the class and def first and the go to the main training loop. refer to what you donno about one specific def.

2. sigmoid, softmax, cross entropy
- sigmoid: non-linear activation function. 
- softmax: scale the vector contains sigmoid values into (0,1).
- cross entropy: loss function. ```- y * log p(y)```, label X the log of the probability of this label 

# 2019-01-02

1. [numpy_kafka_sentence_generate_RNN](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_RNN.md)

review and fix some code error. this tutorial has no batch training. inputs is a sentence with 25 words.



# 2019-01-01

i will continue my journey with deep leaerning and of course NLP !!!

more effiently focus on my own business.

sleep early, get up early and excise more.


# 2018-12-20

## 1. Kaggle download dataset

Reference: [tensorflow_official_word2vec_skipgram.ipynb](https://github.com/davidkorea/tensorflow/blob/master/tensorflow_official_word2vec_skipgram.ipynb)
```python
import os
import argparse
import sys
from tempfile import gettempdir
from six.moves import urllib

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
# http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
url = ' http://www.fit.vutbr.cz/~imikolov/rnnlm/'

# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
    return local_filename


tgz_filename = maybe_download('simple-examples.tgz', 34869662)
```
```
tgz_filename
# '/tmp/simple-examples.tgz'
os.listdir('/tmp')
# ['.ipython', '.config', '.keras', 'simple-examples.tgz', '.local', '.cache']
```
## 2. Kaggle unzip .tgz file in temp
```python
import tarfile
tarobj = tarfile.open(tgz_filename, "r:gz")
for tarinfo in tarobj:
    tarobj.extract(tarinfo.name, r"/tmp")
tarobj.close()
```
A new folder named 'simple-examples' will be created in ```/tmp```, and go to find the .txt file
```python
data_path = '/tmp/simple-examples/data'

with tf.gfile.GFile('/tmp/simple-examples/data/ptb.test.txt', "r") as f:
    print( f.read().replace("\n", "<eos>").split() )
        
```


# 2018-12-19
[tensorflow_RNN_LSTM_MNIST](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_RNN_LSTM_MNIST.md)

# 2018-12-18

1. [numpy_kafka_sentence_generate_LSTM](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_LSTM.md)

2. [numpy_kafka_sentence_generate_RNN](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_RNN.md)

3. [tensorflow_RNN_LSTM_basic](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_RNN_LSTM_basic.md)



# 2018-12-17

https://www.tensorflow.org/api_docs/python/tf/unstack


# 2018-12-16

[tensorflow_RNN_LSTM_MNIST](https://github.com/davidkorea/NLP_201811/blob/master/tensorflow/tensorflow_RNN_LSTM_MNIST.ipynb)

refer: [https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2/full_code.py](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2/full_code.py),[recurrent_network.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)

**Issue**:

**ValueError: Variable RNN/BasicRNNCell/Linear/Matrix already exists, disallowed. Did you mean to set reuse=True in VarScope?**

**Answer**:
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
```
Before excuting code, run ```tf.reset_default_graph()```first.

# 2018-12-14

[Recurrent neural networks and LSTM tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)
![WeChat Image_20181214185217.jpg](https://i.loli.net/2018/12/14/5c137d6081989.jpg)

deep learning from scrathc book by keras - Amazon


# 2018-12-10

[numpy_kafka_sentence_generate_LSTM](https://github.com/davidkorea/NLP_201811/tree/master/RNN_LSTM_GRU/code#numpy_kafka_sentence_generate_lstm)

# 2018-12-06 code for LSTM with numpy
[numpy_kafka_sentence_generate_LSTM.ipynb](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_LSTM.ipynb)

# 2018-12-05 

## 1. Excellent blogs about LSTM
- [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)

- [Materials to understand LSTM](https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1)
## 2.Excellent tutorial for numpy LSTM implementation
- [numpy_lstm](http://blog.varunajayasiri.com/numpy_lstm.html)

# 2018-12-04 Learn_Deep_Learning_in_6_Weeks
## 1. Coding for Create Kafka Sentence RNN - numpy ONLY
1. [numpy_kafka_sentence_generate_RNN.ipynb](https://github.com/davidkorea/NLP_201811/tree/master/RNN_LSTM_GRU/code)
## 2. [Learn Deep Learning in 6 Weeks - Week 3 - Recurrent Networks](https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks#week-3---recurrent-networks)
1. [Generating Text using an LSTM Network No libraries](https://github.com/llSourcell/LSTM_Networks)
2. [Visual Analysis for State Changes in RNNs](https://github.com/HendrikStrobelt/LSTMVis)


# 2018-12-03 Learn_Deep_Learning_in_6_Weeks
## 1. [Learn Deep Learning in 6 Weeks - Week 3 - Recurrent Networks](https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks#week-3---recurrent-networks)

1. [Create Kafka Sentence RNN - numpy ONLY](https://github.com/llSourcell/recurrent_neural_network/blob/master/RNN.ipynb)

2. [Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN)](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)
3. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
## 2. [Grokking-Deep-Learning](https://github.com/iamtrask/Grokking-Deep-Learning)
1. [Manning online book](https://www.manning.com/books/grokking-deep-learning?a_aid=grokkingdl&a_bid=32715258)



# 2018-11-30 
[Recurrent neural networks and LSTM tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)


# 2018-11-29 
## 1. GPU Checking
1. Check use GPU or not
```python
import tensorflow as tf
tf.test.gpu_device_name()
=>out[1]: '/device:GPU:0' # use GPU
=>out[2]: '' # do not use GPU
```
2. Which GPU is using
```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```
```
[name: "/device:CPU:0"
 device_type: "CPU"
 memory_limit: 268435456
 locality {
 }
 incarnation: 12748258960639313348, name: "/device:GPU:0"
 device_type: "GPU"
 memory_limit: 11281304781
 locality {
   bus_id: 1
   links {
   }
 }
 incarnation: 13629897258954568467
 physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"]
```
Reference： [First Google Colaboratory & Basic Tutorial](https://github.com/davidkorea/google_colaboratory/blob/master/first_colab.ipynb)
## 2. [Recurrent neural networks and LSTM tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)
# 2018-11-28
1. [TensorFlow Eager tutorial](http://adventuresinmachinelearning.com/tensorflow-eager-tutorial/)

http://adventuresinmachinelearning.com/keras-eager-and-tensorflow-2-0-a-new-tf-paradigm/



# 2018-11-27 [NLP_201811/RNN_LSTM_GRU](https://github.com/davidkorea/NLP_201811/tree/master/RNN_LSTM_GRU)

## 1. ML Lecture 21-1: Recurrent Neural Network (Part I) - HungyiLee

1. video: [https://www.youtube.com/watch?v=xCGidAeyS4M&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=30](https://www.youtube.com/watch?v=xCGidAeyS4M&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=30)
2. pdf: [http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html)

## 2. deeplearning.ai Sequence Model - AndrewNg

1. video & pdf: [https://mooc.study.163.com/learn/2001280005?tid=2001391038#/learn/content](https://mooc.study.163.com/learn/2001280005?tid=2001391038#/learn/content)

## 3. Recurrent Neural Network - Wanmen

1. video: []()
2. pdf: [https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/RecurrentNeuralNetwork_wanmen.pdf](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/RecurrentNeuralNetwork_wanmen.pdf)

# 2018-11-26

## 1. [Introduction to TensorBoard and TensorFlow visualization](http://adventuresinmachinelearning.com/introduction-to-tensorboard-and-tensorflow-visualization/)

![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/03/TensorFlow-data-flow-graph.gif)

- [Python TensorFlow Tutorial – Build a Neural Network](http://adventuresinmachinelearning.com/python-tensorflow-tutorial/)

a=(b+c)∗(c+2)
![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/03/Simple-graph-example.png)

This may seem like a silly example – but notice a powerful idea in expressing the equation this way: **two of the computations (d=b+c and e=c+2) can be performed in parallel.**  By splitting up these calculations across CPUs or GPUs, this can give us significant gains in computational times.


- [Weight initialization tutorial in TensorFlow](http://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/)

## 2. [Natural Language Processing is Fun! How computers understand Human Language](https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e)
  
- [Parsing English in 500 Lines of Python](https://explosion.ai/blog/parsing-english-in-python)
![](https://cdn-images-1.medium.com/max/800/1*onc_4Mnq2L7cetMAowYAbA.png)

- [State-of-the-art neural coreference resolution for chatbots](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30)
![](https://cdn-images-1.medium.com/max/800/1*vGPbWiJqQA65GlwcOYtbKQ.png)
