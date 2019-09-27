
Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started [using](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/) such a model in production in late 2016. These models are explained in the two pioneering papers ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)).

I found, however, that understanding the model well enough to implement it requires unraveling a series of concepts that build on top of each other. I thought that a bunch of these ideas would be more accessible if expressed visually. That's what I aim to do in this post. You'll need some previous understanding of deep learning to get through this post. I hope it can be a useful companion to reading the papers mentioned above (and the attention papers linked later in the post).


A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images...etc) and outputs another sequence of items. A trained model would work like this:
<video width="100%" height="auto" loop autoplay controls>
  ![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/1.gif)
  Your browser does not support the video tag.
</video>



<!--more-->

<br />

In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words:


![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/2.gif)



## Looking under the hood

Under the hood, the model is composed of an <span class="encoder">encoder</span> and a <span class="decoder">decoder</span>.

The <span class="encoder">encoder</span> processes each item in the input sequence, it compiles the information it captures into a vector (called the <span class="context">context</span>). After processing the entire input sequence, the <span class="encoder">encoder</span> send the <span class="context">context</span>  over to the <span class="decoder">decoder</span>, which begins producing the output sequence item by item.

![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/3.gif)



<br />

The same applies in the case of machine translation.


![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/4.gif)



The <span class="context">context</span>  is a vector (an array of numbers, basically) in the case of machine translation. The <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  tend to both be recurrent neural networks (Be sure to check out Luis Serrano's [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg) for an intro to RNNs).

<div class="img-div" markdown="0">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/context.png" />
    The <span class="context">context</span>  is a vector of floats. Later in this post we will visualize vectors in color by assigning brighter colors to the cells with higher values.
</div>

You can set the size of the <span class="context">context</span>  vector when you set up your model. It is basically the number of hidden units in the <span class="encoder">encoder</span> RNN. These visualizations show a vector of size 4, but in real world applications the <span class="context">context</span> vector would be of a size like 256, 512, or 1024.

<br />

By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state. The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called "[word embedding](https://machinelearningmastery.com/what-are-word-embeddings/)" algorithms. These turn words into vector spaces that capture a lot of the meaning/semantic information of the words (e.g. [king - man + woman = queen](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)).

<br />

<div class="img-div" markdown="0">
    <img src="https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/embedding.png" />
    We need to turn the input words into vectors before processing them. That transformation is done using a <a href="https://en.wikipedia.org/wiki/Word_embedding">word embedding</a> algorithm. We can use <a href="http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/">pre-trained embeddings</a> or train our own embedding on our dataset. Embedding vectors of size 200 or 300 are typical, we're showing a vector of size four for simplicity.
</div>

Now that we've introduced our main vectors/tensors, let's recap the mechanics of an RNN and establish a visual language to describe these models:


![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/RNN_1.gif)


<br />

The next RNN step takes the second input vector and hidden state #1 to create the output of that time step. Later in the post, we'll use an animation like this to describe the vectors inside a neural machine translation model.

<br />

In the following visualization, each pulse for the <span class="encoder">encoder</span> or <span class="decoder">decoder</span>  is that RNN processing its inputs and generating an output for that time step. Since the <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  are both RNNs, each time step one of the RNNs does some processing, it updates its <span class="context">hidden state</span>  based on its inputs and previous inputs it has seen.

Let's look at the <span class="context">hidden states</span>  for the <span class="encoder">encoder</span>. Notice how the last <span class="context">hidden state</span>  is actually the <span class="context">context</span>  we pass along to the <span class="decoder">decoder</span>.



![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/5.gif)



<br />

The <span class="decoder">decoder</span>  also maintains a <span class="decoder">hidden states</span>  that it passes from one time step to the next. We just didn't visualize it in this graphic because we're concerned with the major parts of the model for now.


Let's now look at another way to visualize a sequence-to-sequence model. This animation will make it easier to understand the static graphics that describe these models. This is called an "unrolled" view where instead of showing the one <span class="decoder">decoder</span>, we show a copy of it for each time step. This way we can look at the inputs and outputs of each time step.

![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/6.gif)


<br />



# Let's Pay Attention Now
The <span class="context">context</span>  vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). These papers introduced and refined a technique called "Attention", which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.


<img src="https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/attention.png" />

<div class="img-div" markdown="0">
    At time step 7, the attention mechanism enables the <span class="decoder">decoder</span>  to focus on the word "étudiant" ("student" in french) before it generates the English translation. This ability to amplify the signal from the relevant part of the input sequence makes attention models produce better results than models without attention.
</div>

<br />

Let's continue looking at attention models at this high level of abstraction. An attention model differs from a classic sequence-to-sequence model in two main ways:

First, the <span class="encoder">encoder</span> passes a lot more data to the <span class="decoder">decoder</span>. Instead of passing the last hidden state of the encoding stage, the <span class="encoder">encoder</span> passes _all_ the <span class="context">hidden states</span>  to the <span class="decoder">decoder</span>:

![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/7.gif)



<br />

Second, an attention <span class="decoder">decoder</span>  does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the <span class="decoder">decoder</span>  does the following:

 1. Look at the set of encoder <span class="context">hidden states</span>  it received -- each <span class="context">encoder hidden states</span>  is most associated with a certain word in the input sentence
 1. Give each <span class="context">hidden states</span>  a score (let's ignore how the scoring is done for now)
 1. Multiply each <span class="context">hidden states</span>  by its softmaxed score, thus amplifying <span class="context">hidden states</span>  with high scores, and drowning out <span class="context">hidden states</span>  with low scores


![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/attention_process.gif)


<br />
<br />

This scoring exercise is done at each time step on the <span class="decoder">decoder</span> side.

Let us now bring the whole thing together in the following visualization and look at how the attention process works:

1. The attention decoder RNN takes in the embedding of the <span class="embedding">\<END\></span> token, and an <span class="decoder">initial decoder hidden state</span>.
1. The RNN processes its inputs, producing an output and a <span class="decoder">new hidden state</span> vector (<span class="decoder">h</span><span class="step_no">4</span>). The output is discarded.
1. Attention Step: We use the <span class="context">encoder hidden states</span> and the <span class="decoder">h</span><span class="step_no">4</span> vector to calculate a context vector (<span class="step_no">C</span><span class="decoder">4</span>) for this time step.
1. We concatenate <span class="decoder">h</span><span class="step_no">4</span> and <span class="step_no">C</span><span class="decoder">4</span> into one vector.
1. We pass this vector through a <span class="ffnn">feedforward neural network</span> (one trained jointly with the model).
1. The <span class="logits_output">output</span> of the feedforward neural networks indicates the output word of this time step.
1. Repeat for the next time steps

![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/attention_tensor_dance.gif)


<br />
<br />

This is another way to look at which part of the input sentence we're paying attention to at each decoding step:


![](https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/9.gif)


Note that the model isn't just mindless aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:


<div class="img-div" markdown="0">
<img src="https://github.com/davidkorea/NLP_201811/blob/master/Neural_Machine_Translation_seq2seq/README/attention_sentence.png" />
    You can see how the model paid attention correctly when outputing "European Economic Area". In French, the order of these words is reversed ("européenne économique zone") as compared to English. Every other word in the sentence is in similar order.
</div>


<br />

If you feel you're ready to learn the implementation, be sure to check TensorFlow's [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).


