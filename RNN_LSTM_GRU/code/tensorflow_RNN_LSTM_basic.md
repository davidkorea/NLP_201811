
# tensorflow_RNN_LSTM_basic
https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/tensorflow_RNN_LSTM_basic.ipynb


(https://i.loli.net/2019/01/04/5c2f0a2a35b04.png)

与[numpy RNN tutorail](https://github.com/davidkorea/NLP_201811/blob/master/RNN_LSTM_GRU/code/numpy_kafka_sentence_generate_RNN.md)不同, 由于此处不需要生成文本text generate，所以hidden计算后，无需进行softmax。所以hidden直接输出为y。

因此，此处2 timestep的例子中，将前面的输出y直接传递给后面来计算，原因就是此处hidden就是y

output_seqs, states = tf.contrib.rnn.static_rnn(cell=basic_cell, inputs=[x0,x1], dtype=tf.float32)
  - output_seqs：每个timestep的输出结果
  - states：final states of the network，应该就是hidden吧？就是memory？


-----

