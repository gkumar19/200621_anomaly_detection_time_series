# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:12:26 2020

@author: Gaurav
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% read data
df = pd.read_csv('S&P_500_Index_Data.csv', parse_dates=['date'])

#%% plot the data
sns.relplot(x='date', y='close', data=df, kind='line')

#%% convert data into sequence
from sklearn.model_selection import train_test_split
X = df['close'].values[:,None]
y = df['close'].values[:,None]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_X.fit(X_train)
scale_y = StandardScaler()
scale_y.fit(y_train)

def raw_to_sequence(X, window):
    len_X = X.shape[0]
    sequence_array = []
    for i in range(len_X- window + 1):
        sequence_array.append(X[i:i+window,:].reshape(1,window,-1))
    return np.concatenate(sequence_array, axis=0)

WINDOW = 30
X_train_sc_wn = raw_to_sequence(scale_X.transform(X_train), WINDOW)
X_test_sc_wn = raw_to_sequence(scale_X.transform(X_test), WINDOW)
y_train_sc_wn = raw_to_sequence(scale_y.transform(y_train), WINDOW)
y_test_sc_wn = raw_to_sequence(scale_y.transform(y_test), WINDOW)

#%% make tensorflow model
hparams = {'lstm_0': 32,
           'dropout_1': 0.2,
           'lstm_3': 32,
           'dropout': 0.2}

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, GRU
model = Sequential([GRU(hparams['lstm_0'],input_shape=(WINDOW,1)),
                   Dropout(hparams['dropout_1']),
                   RepeatVector(WINDOW),
                   GRU(hparams['lstm_3'], return_sequences=True),
                   Dropout(hparams['dropout']),
                   TimeDistributed(Dense(1))])
model.compile(optimizer='adam', loss='mse')
model.summary()

#%% train model
history = model.fit(X_train_sc_wn, y_train_sc_wn, epochs=80,
                    batch_size=1024, validation_split=0.2, shuffle=False)
#%% generate mae
from sklearn.metrics import mean_squared_error
mae = np.abs(y_train_sc_wn - model.predict(X_train_sc_wn))
mae = np.mean(mae, axis=(-2, -1))

#%% plot mae
plt.hist(mae, bins=30)
print(np.percentile(mae, 99))
#%% selecting threshold
anamoly = mae > np.percentile(mae, 98)
anamoly = np.concatenate([np.array([False]*(WINDOW-1)) , anamoly], axis=0)

#%% plot anamoly
df_anamoly = pd.DataFrame()
df_anamoly['close'] = X_train.flatten()
df_anamoly['anamoly'] = anamoly
df_anamoly['date'] = np.arange(len(df_anamoly))
predict = scale_y.inverse_transform(np.mean(model.predict(X_train_sc_wn), axis=1).flatten())
predict = np.concatenate([np.array([0]*(WINDOW-1)) , predict], axis=0)
df_anamoly['close_predict'] = predict
sns.relplot(x='date', y='close', data=df_anamoly, kind='line')
plt.plot(df_anamoly['date'], df_anamoly['close_predict'], color='k', label='predicted')
df_anamoly = df_anamoly[df_anamoly['anamoly'] == True]
plt.scatter(df_anamoly['date'], df_anamoly['close'], color='red', label='anamoly')
plt.legend()
