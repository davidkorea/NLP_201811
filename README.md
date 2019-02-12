# NLP_201811

# 2019-02-12
## 1. Error - https://github.com/kuza55/keras-extras/issues/7#issuecomment-447235795
```
InvalidArgumentError: Incompatible shapes: [21504] vs. [1024,21]

[[{{node metrics_1/acc/Equal}} = Equal[T=DT_FLOAT, 
	 _device="/job:localhost/replica:0/task:0/device:CPU:0"]
	 (metrics_1/acc/Reshape, metrics_1/acc/Cast)]]
```
1. I get this error only when run my code on a GPU node (Tesla k80)
2. I do not get the error for batch_size = 1
3. I do not get the error when I do not use metrics=['accuracy'] in compile.
4. I get the error only for some particular architecture
5. All the problems reported above have problems with arrays of the same dimensionality [n1,n2]
  vs [m1,m2] but my  case is [n] vs [n/r, r]

# 2019-02-11
1. https://github.com/modin-project/modin
```
DaviddeMBP:~ david$ source activate tensorflow
(tensorflow) DaviddeMBP:~ david$ pip install modin
```
2. https://carbon.now.sh

3. https://nteract.io/ download desktop app
4. https://www.kite.com/


# 2019-02-10
first day saty at home alone

# 2019-02-09
天宝
# 2019-02-08
드라마 데이
# 2019-02-07
외할아버지 생신 

# 2019-02-06
힘들어,드람 ㄱㅊ

# 2019-02-05
힘들다ㅠㅠ 

# 2019-02-04

come back home.. what to say...

# 2019-02-02 https://www.youtube.com/watch?v=t5qgjJIBy9g


Lets Make a Question Answering chatbot using the bleeding edge in deep learning (Dynamic Memory Network). We'll go over different chatbot methodologies, then dive into how memory networks work, with accompanying code in Keras. 

Code + Challenge for this video:
https://github.com/llSourcell/How_to_make_a_chatbot

Nemanja's Winning Code:
https://github.com/Nemzy/language-translation/blob/master/neural_machine_translation.ipynb

Vishal's Runner up code:
https://github.com/erilyth
