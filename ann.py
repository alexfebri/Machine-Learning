from pandas import read_csv
import pathlib
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv('nopre.csv')
data.head()

y1 = data['TPT']
x1 = data.drop(columns='TPT')
X = x1.values
Y = y1.values

train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)
train_labels = train_dataset.pop('TPT')
test_labels = test_dataset.pop('TPT')

def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()


EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print (example_result)


test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()