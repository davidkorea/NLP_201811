# NLP_201811

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
