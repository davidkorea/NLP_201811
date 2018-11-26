안녕하세요,  여러분. 저는 조니입니다. 저는 중구구에서 













# tensorflow
andrew wu

# Issue 8 LSTM
![](https://i.loli.net/2018/11/24/5bf91df88d0e8.png)
![](https://i.loli.net/2018/11/24/5bf91dfb11a45.png)


![](https://i.loli.net/2018/11/24/5bf91dd93522d.png)

![](https://i.loli.net/2018/11/24/5bf91dec59515.png)


![](https://i.loli.net/2018/11/24/5bf91df89b581.png)


![](https://i.loli.net/2018/11/24/5bf91dfc16525.png)

![](https://i.loli.net/2018/11/24/5bf91e01ba3f0.png)



# Issue 7 - tf.nn.nce_loss()

```python

Signature: tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, 
                            sampled_values=None, remove_accidental_hits=False, 
                            partition_strategy='mod', name='nce_loss')
```


# Issue 6 - os.makedir('./data') on Kaggle Kernel

1. Kaggle kernel supports create a folder path with code ```os.makedir('./data')```
2. if using a uploaded dataset, it will be saved ```'../input/'```

Refer to notebook [tensorflow_skipgram.ipynb](https://github.com/davidkorea/tensorflow/blob/master/tensorflow_skipgram.ipynb)

![](https://i.loli.net/2018/11/21/5bf4d18f5eca0.png)

Reference: [https://github.com/nishithbsk/tensorflow_tutorials/tree/master/cs224n/word2vec](https://github.com/nishithbsk/tensorflow_tutorials/tree/master/cs224n/word2vec)

# Issue 5 - tf.argmax(vector, 1) same as np.argmax()

> ```tf.argmax(input, axis=None, name=None, dimension=None, output_type=tf.int64)```

- 0：按列计算
- 1：按行计算
```python
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
```

1.  tf.argmax(vector, 1)

```python
test[0] = array([1, 2, 3])  #2
test[1] = array([2, 3, 4])  #2
test[2] = array([5, 4, 3])  #0
test[3] = array([8, 7, 2])  #0
```
2.  tf.argmax(vector, 0)

```python
test[0] = array([1, 2, 3])
test[1] = array([2, 3, 4])
test[2] = array([5, 4, 3])
test[3] = array([8, 7, 2])
# output   :  # [3, 3, 1]   
```

# Issue 4 - Open Tensorborad in Kaggle Kernel

1. open tensorboard locally

```python
writer = tf.summary.FileWriter('./graph1', sess.graph)
writer.close()
sess.close()

!tensorboard --logdir='./graph1'
```
2. open tensorboard on Kaggle Kernel

```python
# At first in settings, Make sure that Internet option is set to "Internet Connected"
# After executing this cell, there will come a link below, open that to view your tensor-board

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
LOG_DIR = './graph1' # Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

![20181119162041](https://user-images.githubusercontent.com/26485327/48691823-4cb2f680-ec17-11e8-8f57-f064b65938b0.png)

# Issue 1 - MAC install tensorflow

1. Install pip3

```$ python3 -m easy_install pip```
```
Searching for pip                                                                                         
Best match: pip 10.0.1           
Adding pip 10.0.1 to easy-install.pth file         
Installing pip script to /usr/local/bin                                                                   
Installing pip3 script to /usr/local/bin                                                                  
Installing pip3.6 script to /usr/local/bin     

Using /usr/local/lib/python3.6/site-packages                                                              
Processing dependencies for pip                                                                           
Finished processing dependencies for pip                                                                  
```
2. Install tensorflow

```$ pip3 install tensorflow```

# Issue 2 - Win7 install tensorflow

1. Re-install to the path without space

```$ conda create -n tensorflow pip python=3.5```
```
WARNING: A space was detected in your requested environment path
'e:\Program Files\Anaconda3\envs\tensorflow'
Spaces in paths can sometimes be problematic.
Solving environment: failed
```

2. Do as the official guidebook - [Installing with Anaconda](https://www.tensorflow.org/install/install_windows?hl=ko)

```$ C:> conda create -n tensorflow pip python=3.5 ```

```$ C:> activate tensorflow```

```$ (tensorflow)C:> pip install --ignore-installed --upgrade tensorflow ```


3. Warnings that could ignore?
```
>>> import tensorflow as tf
>>> sess = tf.Session()

2018-07-17 00:46:16.481492: I T:\src\github\tensorflow\tensorflow\core\platform\
cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow bi
nary was not compiled to use: AVX2

>>> hello = tf.constant('hello, tensor flow')
>>> print(sess.run(hello))
b'hello, tensor flow'
```
  reference:
  [Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)
  
  
 # Issue 3 - Win10 import error
 
 **Unfixed**
 
 ```
 >>> import tensorflow as tf
Traceback (most recent call last):
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 14, in swig_import_helper
    return importlib.import_module(mname)
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 985, in _gcd_import
  File "<frozen importlib._bootstrap>", line 968, in _find_and_load
  File "<frozen importlib._bootstrap>", line 957, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 666, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 577, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 938, in create_module
  File "<frozen importlib._bootstrap>", line 222, in _call_with_frames_removed
ImportError: DLL load failed: DLL 초기화 루틴을 실행할 수 없습니다.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 17, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 16, in swig_import_helper
    return importlib.import_module('_pywrap_tensorflow_internal')
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: No module named '_pywrap_tensorflow_internal'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\__init__.py", line 22, in <module>
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 74, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 14, in swig_import_helper
    return importlib.import_module(mname)
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 985, in _gcd_import
  File "<frozen importlib._bootstrap>", line 968, in _find_and_load
  File "<frozen importlib._bootstrap>", line 957, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 666, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 577, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 938, in create_module
  File "<frozen importlib._bootstrap>", line 222, in _call_with_frames_removed
ImportError: DLL load failed: DLL 초기화 루틴을 실행할 수 없습니다.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 17, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 16, in swig_import_helper
    return importlib.import_module('_pywrap_tensorflow_internal')
  File "C:\Users\xerox\Anaconda3\envs\tensorflow\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: No module named '_pywrap_tensorflow_internal'


Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/install_sources#common_installation_problems

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
>>>
 ```
