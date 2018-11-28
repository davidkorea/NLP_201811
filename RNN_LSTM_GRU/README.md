> **References about this ariticle**
> 1. ML Lecture 21-1: Recurrent Neural Network (Part I) - HungyiLee
> 2. deeplearning.ai Sequence Model - AndrewNg
> 3. Recurrent Neural Network - Wanmen

# 1. Why Sequence Model / RNN

For sequence data (time, vidoe, radio, ...), why don't we use the fully connected nerual network? This is an exanple to make it more clearily to should we select a model that fits sequence data well.

- Slot Filling

Slot filling is generally used in the robot customer service filed to automatically answer the specific question the customer raised by some key words. For example, the flight ticket booking app may get one message "I would like to arrive Taipei on Nov. 2nd". the program should identify which word belongs to which slot.

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/slotfilling.png?raw=true)

BUT, the problem is that how should we tell "leave" and "arrive" apart. "place of arrive" OR "place of departure"

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

It could be understanded easily by the pic above, we set all the weights as 1 and the init memorn cell is 0. therefore all the outputs will be calculated. 

Also, the order of the sequence input is important and the output will totally different if the order changed. This also demonstrated that the current output has an dependency with the previous input just as the pic shows below.

<p align="center">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNorder.png">
</p>

In this way, before we decide whick slot to fill, we can read the previous infomation first and then make the dicision. "place of arrive" or "place of departure" is dependent on "arrive" or "leave".

## 2.2 Algorithm
### 2.2.1 Forward Propagation
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNnotation.png?raw=true)
**same color belongs to same part of equation in simplified notation**

Through the mathmatic notation and equation we can find that:
1. hidden layer: 
    - previous hidden output -> transpose matrix, 
    - input -> transpose matrix, 
    - bias
    
    add up and pass to a nolinear activation function
    
2. output: 
    - current hidden output -> transpose matrix,
    - bias
    
    add up and pass to a nolinear activation function, ```sigmoid``` for binary classification and ```softmax``` for multi-classification.

### 2.2.2 Backward Propagation
![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBP.png?raw=true)

![](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/RNNBP2.png?raw=true)

<p align="center"><img src="https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/README/BPgradient.png"></p>

The gradient equation which is hidden layer weights gradient with respect to the 3rd output L3. Because of the "∏(...)" makes it hard to pass the gradient to the very beginning when the large dimentions of the hidden layer.

After the sigmoid function, the hidden layer will give an ouput in the range(0, 1).
- 1.00 ^ 1000 = 1
- 1.01 ^ 1000 = 20000
- 0.99 ^ 1000 = 0
- 0.01 ^ 1000 = 0

we can figure out that even the small difference 1.01, 0.99 will make a huge completely diffenent output after many layers backward propagation. whereas even there's a big difference 0.99, 0.01 will give the same output 0. we call this Gradient Vanish.

And from the Loss Function surface plot we see that the cliff will make it hard to train even after many epochs
