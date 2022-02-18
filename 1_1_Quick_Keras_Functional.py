#!/usr/bin/env python
# coding: utf-8

# # Keras로 Linear&Logistic Regression 맛보기!
# 
# ### Kaggle Link for exercise :
# * [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) : 
# 
# #### 실습목표<br>
# 1. keras의 모델링 아이디어를 이해한다.
# 2. 모든 코드를 이해한다.

# ## Quick Linear Regression!

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np


# In[2]:


x = np.array(range(0,20)) 
y = x * 2 -1

print(x)
print(y)


# In[3]:


x.shape, y.shape


# In[4]:


# 혹시 이미 그려둔 그래프가 있다면 날려줘!
keras.backend.clear_session()

# 레이어들을 사슬로 연결하 듯이 연결!
il = keras.layers.Input(shape=(1,))
ol = keras.layers.Dense(1)(il)

# 모델의 시작과 끝을 지정
model = keras.models.Model(il, ol)

# 컴파일 해주렴
model.compile(loss = 'mse', optimizer = 'adam')
#metrics에서는 선형회귀에서 선언 xxx


# In[5]:


# 데이터를 넣어서 학습시키자!
model.fit(x, y, epochs=10, verbose=1)


# In[6]:


# 결과 출력해줘!
print(y)
print(model.predict(x).reshape(-1,) )


# ## Now, Your turn!

# In[7]:


import tensorflow as tf
from tensorflow import keras

import numpy as np


# In[8]:


x = np.array(range(0,20)) 
y = x * (-3) + 10

print(x)
print(y)


# In[9]:


x.shape, y.shape


# In[10]:


## Functional API

# 1.세션 클리어 
keras.backend.clear_session()

# 2. 레이어 사슬 연결
il = keras.layers.Input(shape=(1,))
ol = keras.layers.Dense(1)(il)

# 3. 모델의 시작과 끝을 지정 
model=keras.models.Model(inputs=input_layer, outputs=output_layer)

# 4.컴파일
model.compile(loss='mse',optimizer='adam')


# In[ ]:


model.fit(x,y, epochs=10, verbose=1)


# In[ ]:


# 결과 출력해줘!
print(y)
print(model.predict(x).reshape(-1,) )


# ## Quick Logistic Regression!

# In[ ]:


import tensorflow as tf
from tensorflow import keras

import numpy as np


# In[ ]:


x = np.array(range(0,20)) 
y = np.array([0]*10 + [1]*10)

print(x)
print(y)


# In[ ]:


# 혹시 이미 그려둔 그래프가 있다면 날려줘!
keras.backend.clear_session()

# 레이어들을 사슬로 연결하 듯이 연결!
input_layer = keras.layers.Input(shape=(1,))
output_layer = keras.layers.Dense(1, activation='sigmoid')(input_layer)

# 모델의 시작과 끝을 지정
model = keras.models.Model(inputs=input_layer, outputs=output_layer)


# 컴파일 해주렴
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])


# In[ ]:


# 데이터를 넣어서 학습시키자!
model.fit(x, y, epochs=10, verbose=1)

# 결과 출력해줘!
print(y)
print(model.predict(x).reshape(-1,) )


# ## Now, Your turn!

# In[ ]:





# In[ ]:


x = np.array(range(0,40)) 
y = np.array([0]*20 + [1]*20)
print(x)
print(y)


# In[ ]:





# In[ ]:





# In[ ]:




