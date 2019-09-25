
> **References about this ariticle**
> 1. ML Lecture 21-1: Recurrent Neural Network (Part I) - HungyiLee
> 2. deeplearning.ai Sequence Model - AndrewNg
> 3. Recurrent Neural Network - Wanmen
> 4. Medium-[Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714) - Shi Yan
> 5. Medium-[Materials to understand LSTM](https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1) - Shi Yan

# 1. Why Sequence Model / RNN

For sequence data (time, vidoe, radio, ...), why don't we use the fully connected nerual network? This is an exanple to make it more clearily to should we select a model that fits sequence data well.

- Slot Filling

Slot filling is generally used in the robot customer service field to automatically answer the specific question the customer raised by catching some key words. For example, the flight ticket booking app may get one message "I would like to arrive Taipei on Nov. 2nd". the program should identify which word belongs to which slot.

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/slotfilling.png?raw=true)

BUT, the problem is that how should we tell "leave" and "arrive" apart. "place of arrive" OR "place of departure".

This the key point that we should use the sequence model / RNN to deal with this kind of question that the current has a strong dependency with the previous input data.

- Named Entity Recognition

This is another example to use RNN.

He said, “Teddy Roosevelt was a great President.”
He said, “Teddy bears are on sale!”

Will "Teddy" be recognized as a people name or a 장난감?

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/NamedEntityRecognition.png?raw=true)

-----

# 2. RNN

Actually, the only difference between fully connected layer and RNN is that each neuron in the hidden layer will pass its value to the next neuron.
## 2.1 Model Architecture
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNN.png?raw=true)

It could be understanded easily by the pic above, we set all the weights as 1 and the init memory cell is 0. therefore all the outputs will be calculated. 

Also, the order of the sequence input is important and the **output will totally different if the order changed**. This also demonstrated that the current output has an dependency with the previous input just as the pic shows below. **Outpus depends on the input order of sequence, this can also be understood as a Memory, different previous input has different memory**

<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNorder.png">
</p>

In this way, before we decide which slot to fill, we can read the previous infomation first and then make the dicision. "place of arrive" or "place of departure" is dependent on "arrive" or "leave".

## 2.2 Algorithm
### 2.2.1 Forward Propagation
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNnotation.png?raw=true)
**same color belongs to same part of equation in simplified notation**

Through the mathmatic notation and equation we can find that:
1. input: ```np.dot(Wax, x)```
    - input vector -> transpose matrix ```Wax```

2. hidden layer: ```np.dot(Waa, h[t-1]) + ba```
    - previous hidden output -> transpose matrix ```Waa```, 
    - bias ```ba```
    
    add up and pass to a nolinear activation function ```h[t] = np.tanh( np.dot(Waa,x) + np.dot(Waa,h[t-a])+ba )```.
    
3. output: ```dot.np(Wah, h[t]) + by```
    - current hidden output -> transpose matrix ```Way```,
    - bias ```by```
    
    add up and pass to a nolinear activation function, ```sigmoid``` for binary classification and ```softmax``` for multi-classification.

### 2.2.2 Backward Propagation
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBP.png?raw=true)

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBP2.png?raw=true)

- 如右下角图所示，假设输入为[1, 0, 0, ..., 0], 输入和输出转换矩阵为1， hiddle的转换矩阵为w，经过1000次计算后，最后的结果是w^1000次方。
    - 因此如果w = 1 那么1的1000次方还是1，但如果w变化一点点为1.01， 那么1.01的1000次方约为20000
    - 再者，当w=0.99 和w=0.01，w本身差异很大，但是经过1000次方计算后，结果都为0

<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/BPgradient.png" width="390" height="130">
</p>

The gradient equation which is hidden layer weights' gradient with respect to the 3rd output L3. Because of the "∏(...)" makes it hard to pass the gradient to the very beginning when there's a large dimentions of the hidden layer.(once 0 occurs all gradient will be 0)

After the sigmoid function, the hidden layer will give an ouput in the range(0, 1).
- 1.00 ^ 1000 = 1
- 1.01 ^ 1000 = 20000
- 0.99 ^ 1000 = 0
- 0.01 ^ 1000 = 0

we can figure out that:
- even a small difference 1.01, 0.99 will make a huge completely diffenent output after many layers backward propagation. 
- whereas 
- even a big difference 0.99, 0.01 will give the same output 0. we call this Gradient Vanish.

And from the Loss Function surface plot we see that the cliff will make it hard to train even after many epochs.

# 2.3 RNN  Code Implementation
## 2.3.1 numpy_kafka_sentence_generate_RNN
- code
    [numpy_kafka_sentence_generate_RNN](https://github.com/davidkorea/NLP_201811/tree/master/RNN_LSTM_GRU/code)
- tutorial
    [llSourcell/recurrent_neural_network/RNN.ipynb](https://github.com/llSourcell/recurrent_neural_network/blob/master/RNN.ipynb)
## 2.3.2 keras_hongloumeng_generate_RNN
- code
    [keras_hongloumeng_RNN.ipynb](https://github.com/davidkorea/google_colaboratory/blob/master/keras_hongloumeng_RNN.ipynb)
- tutorial
    [Text Generate toturial - Hongloumeng](https://github.com/davidkorea/deeplearning/blob/master/递归神经网络(1).ipynb)

-----

# 3. LSTM

## 3.1 Algorithm

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTM1.png)

---
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTM2.png)

---
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTM3.png)

The last pic as is shown above, input x -> transpose matrix -> z, z_i, z_o, z_f vecters.

Each dimention of the vector will control one LSTM neuron which is the one dimention of hidden layer. And Each LSTM neuron need 4 inputs.

That is to say, as showed below, input vector dim is the same as hidden layer dim as also teh same as output softmax dim in the situation of many-to-many networks. 

<p align="center">
    <img src="http://adventuresinmachinelearning.com/wp-content/uploads/2017/10/LSTM-many-to-many-classifier-3.png" width="400" height="400">
</p>

---
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTM6.png)

---
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMstepbox.png)

---
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTM2graph.png)


For LSTM, there are 4 inputs and 1 output.
1. **input**: word2vec x -> transpose matrix = z
2. **input gate**: word2vec x -> transpose matrix_i = z_i
3. **forget gate**: word2vec x -> transpose matrix_f = z_f
4. **output gate**: word2vec x -> transpose matrix_o = z_o

The 4 transpose matrix are different and will be trained by model through BP.

If the previous infomation recognized by the model is important that learned through the training set. the infomation can be passed to current by setting the forget gate control signal = 1 continuously. Generally, the forget gate is always on (remember) and the input,output gate are always off with no input according to the init weight/transpose matrix shows above.

Moreover, the upgraded version **LSTM_peephole**. Not only input word2vec input x, but also togethering with previous output,hidden(memory) to generate the gate control signal.
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeephole.png)

1. **input**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix = z
2. **input gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_i = z_i
3. **forget gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_f = z_f
4. **output gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_o = z_o

What we have talked about is ONE hidden layer LSTM network. Obviously, RNN/LSTM could be deep, BUT less than 3 hidden layers in general for huge ammount of parameters will be hard to train due to the limited computional power as well as huge time consuming.


> Thanks for the Excellent blogs, show us the exact diagram about LSTM with peephole.
> - [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
> - [Materials to understand LSTM](https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1)

<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeepholebad.png">
    <p align="center">
        <em>Error fixed in red circle</em>
    </p>
</p>

> People never judge an academic paper by those user experience standards that they apply to software. If the purpose of a paper were really promoting understanding, then most of them suck.

-----
<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeepholemid.png">
    <p align="center">
        <em>Blue lines are peephole</em>
    </p>   
    <p align="center">
        <em>3 Dotlines(2 blue & 1 black) are the last/previous memory cell values</em>
    </p>   
    <p align="center">
        <em>Bond lines (black & blue) are weighted connections</em>
    </p>   
    <p align="center">
        <em>Thin lines are un-weighted connections</em>
    </p>
</p>

-----
<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeepholegood.png" width="500" height="500">
    <p align="center">
        <em>3 Red dot lines are last/previous memory cell values</em>
    </p>
</p>

-----
<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeepholecgfix.png" width="600" height="500">
    <p align="center">
        <em>3 Light blue arrows are last/previous memory cell values</em>
    </p>    
</p>

## 3.2 LSTM Code Implementation
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMvanillaBPcg.png)
- code
    [numpy_kafka_sentence_generate_RNN](https://github.com/davidkorea/NLP_201811/tree/master/RNN_LSTM_GRU/code)
- tutorial
    [Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html)
