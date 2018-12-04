- [keras_hongloumeng_RNN.ipynb](https://github.com/davidkorea/google_colaboratory/blob/master/keras_hongloumeng_RNN.ipynb)

- [Text Generate toturial - Hongloumeng](https://github.com/davidkorea/deeplearning/blob/master/递归神经网络(1).ipynb)

> **References about this ariticle**
> 1. ML Lecture 21-1: Recurrent Neural Network (Part I) - HungyiLee
> 2. deeplearning.ai Sequence Model - AndrewNg
> 3. Recurrent Neural Network - Wanmen

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

# 2. RNN

Actually, the only difference between fully connected layer and RNN is that each neuron in the hidden layer will pass its value to the next neuron.
## 2.1 Model Architecture
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNN.png?raw=true)

It could be understanded easily by the pic above, we set all the weights as 1 and the init memory cell is 0. therefore all the outputs will be calculated. 

Also, the order of the sequence input is important and the output will totally different if the order changed. This also demonstrated that the current output has an dependency with the previous input just as the pic shows below.

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

# 3. LSTM

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

Moreover, the upgraded version LSTM_peephole. Not only input word2vec input x, but also togethering with previous output,hidden(memory) to generate the gate control signal.
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/LSTMpeephole.png)

1. **input**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix = z
2. **input gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_i = z_i
3. **forget gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_f = z_f
4. **output gate**: ( word2vec x, precious_hidden, previous_output ) -> transpose matrix_o = z_o

What we have talked about is ONE hidden layer LSTM network. Obviously, RNN/LSTM could be deep, BUT less than 3 hidden layers in general for huge ammount of parameters will be hard to train due to the limited computional power as well as huge time consuming.
