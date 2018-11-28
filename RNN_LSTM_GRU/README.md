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
