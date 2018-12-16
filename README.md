
# NLP_201811

# 2018-12-16

[tensorflow_RNN_LSTM_MNIST](https://github.com/davidkorea/NLP_201811/blob/master/tensorflow/tensorflow_RNN_LSTM_MNIST.ipynb)

refer: [recurrent_network.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)

**Issue: **
**ValueError: Variable RNN/BasicRNNCell/Linear/Matrix already exists, disallowed. Did you mean to set reuse=True in VarScope?**

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
