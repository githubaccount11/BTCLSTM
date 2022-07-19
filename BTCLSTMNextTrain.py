# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:58:55 2020

@author: User
"""


import pandas as pd
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "BTC"
EPOCHS = 1  # how many passes through our data
BATCH_SIZE = 4  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-200x5-Batch-4-EPOCH-4-BSize-4-{int(time.time())}"



def preprocess_df(df):
    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store 
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels 
    
    return np.array(X), y  # return X and y...and make X a numpy array!


main_df = pd.DataFrame() # begin empty

ratios = ["BTC"]  # the stocks we want to consider
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    print(ratio)
    dataset = f'BTCData/BTC4.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['pct_change', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"pct_change": f"{ratio}_pct_change", "volume": f"{ratio}_volume"}, inplace=True)

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['target'] = main_df[f'{RATIO_TO_PREDICT}_pct_change'].shift(-FUTURE_PERIOD_PREDICT)

main_df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]
#print(last_5pct)
validation_main_df = main_df[(main_df.index >= last_5pct)]
#print(validation_main_df)
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

opt = tf.keras.optimizers.Adam(lr=0.0001)

model = load_model('models\\BTC-60-SEQ-1-PRED-200x5-Batch-4-EPOCH-3-BSize-4-1590621043')

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
)

# Train model
history = model.fit(
    np.array(train_x), np.array(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(np.array(validation_x), np.array(validation_y)),
)

# Save model
model.save("models\\{}".format(NAME))















