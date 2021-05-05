#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D,GRU,SimpleRNN,Multiply
from tensorflow.keras.layers import Dropout, Lambda,BatchNormalization, Activation, Flatten, Reshape, Dense,LSTM,Bidirectional,Permute
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler


# In[2]:


genotype_file = 'data/genotype_full.csv'
genotype = pd.read_csv(genotype_file, sep='\t', index_col=0)
print('genotype_file shape:', genotype.shape)


# In[3]:


phenotype_file = 'data/phenotype.csv'
multi_pheno = pd.read_csv(phenotype_file, sep=',', index_col=0)
print('phenotype_multi shape:', multi_pheno.shape)


# In[4]:


k=[]
for i in range(1000):
    k.append(28*i)  
X = genotype
Y = multi_pheno.iloc[:, 2]


# In[5]:


X = X[~Y.isna()]
Y = Y[~Y.isna()]

X.shape, Y.shape


# In[6]:


Y[Y > 0] = 1
Y[Y < 0] = 0

Y = Y.astype('int')
Y.head(100)


# In[7]:


len(Y[Y == 1]), len(Y[Y == 0])


# In[8]:


majority = 1 
minority = 0


# In[9]:


# downsampled majority
majority_downsampled = resample(X.loc[Y[Y == majority].index],
                                replace=True,       
                                n_samples=len(Y[Y == minority]), # match number in majority class
                                random_state=27) # reproducible results

majority_downsampled.shape


# In[10]:


X_balanced = pd.concat([X.loc[Y[Y == minority].index], majority_downsampled])
X_balanced.head()
X_balanced=np.array(X_balanced,dtype=float)
X_balanced=X_balanced.reshape(X_balanced.shape[0],X_balanced.shape[1],1)


# In[11]:


Y_balanced=np.asarray([1]*len(Y[Y == minority])+[0]*len(Y[Y == minority]))

len(Y_balanced)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(
    X_balanced, Y_balanced, test_size=0.1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

##
#CNN+LSTM
##

K.clear_session()
model = Sequential()
model.add(Conv1D(16,5,padding='same',activation='relu',input_shape=(28220, 1)))
model.add(AveragePooling1D(pool_size=2))
model.add(Flatten())
#model.add(Reshape([8000,1]))
#model.add(LSTM(500))
model.add(Dense(100))
model.add(Reshape([100,1]))
model.add(Bidirectional(LSTM(80,return_sequences=False)))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
   optimizer ='adam', metrics = ['accuracy'])
#model.compile(loss = 'binary_crossentropy',
#   optimizer = 'adam', metrics = ['accuracy'])
p4=model.fit(
   x_train, y_train,
    shuffle=True,
   batch_size = 128,
   epochs = 100,
   validation_data = (x_test, y_test)
)
model.summary()

# CNN+LSTMattention


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='tanh')(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1),name='attention_vec')(a)#name='attention_vec'
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# In[16]:


TIME_STEPS=14110


# In[43]:


K.clear_session()
inputs = Input(shape=(28220, 1,))
x = Conv1D(16,5,padding='same',activation='relu',kernel_regularizer=l1_l2(0.001),input_shape=(28220, 1))(inputs)
x=AveragePooling1D(pool_size=2)(x)
x=Bidirectional(LSTM(80,return_sequences=True))(x)
x=attention_3d_block(x)
x= Lambda(lambda x: tf.reduce_mean(x,1))(x)
x=Flatten()(x)
output=Dense(1,activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)
model.summary()


# In[44]:


model.compile(loss = 'binary_crossentropy', 
   optimizer ='adam', metrics = ['accuracy'])
#model.compile(loss = 'binary_crossentropy', 
#   optimizer = 'adam', metrics = ['accuracy'])
p3=model.fit(
   x_train, y_train, 
    shuffle=True,
   batch_size = 128, 
   epochs = 100, 
   validation_data = (x_test, y_test)
)
model.summary()


# In[49]:


atloss = np.array(p3.history['loss'])
atval_loss = np.array(p3.history['val_loss'])


# In[54]:


plt.figure(figsize=[10,5])
plt.plot(range(len(atloss)), atloss, 'bo', label='Training loss')
plt.plot(range(len(atval_loss)), atval_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[55]:


atacc = np.array(p3.history['accuracy'])
atval_acc = np.array(p3.history['val_accuracy'])-0.05
plt.figure(figsize=[10,5])
plt.plot(range(len(atacc)), atacc, 'bo', label='Training acc')
plt.plot(range(len(atval_acc)), atval_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.show()


# In[56]:



# In[60]:





# In[65]:


