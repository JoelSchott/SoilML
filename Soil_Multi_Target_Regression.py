from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print (tf.__version__)

raw_dataset = pd.read_csv('C:\\Users\\Computer\\Documents\\Example Soil Data.csv')
data_set = raw_dataset.copy()
data_set.pop('x')
data_set.pop('y')

train_dataset = data_set.sample(frac = 0.8, random_state=0)
test_dataset = data_set.drop(train_dataset.index)

#sns.pairplot(train_dataset[['conductivity',  'water content', 'sand', 'clay', 'silt']], diag_kind = 'kde')
#plt.show()

train_stats = train_dataset.describe()
print(train_stats)
train_stats.pop('sand')
train_stats.pop('clay')
train_stats.pop('silt')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset[['sand','clay','silt']].copy()
train_dataset.drop(['sand','clay','silt'], axis = 1, inplace=True)


test_labels = test_dataset[['sand', 'clay', 'silt']].copy()
test_dataset.drop(['sand','clay','silt'], axis = 1, inplace = True)

print(train_dataset)
print(train_labels)

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(3)
        ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss = 'mean_squared_error', optimizer=optimizer, metrics = ['mean_absolute_error', 'mean_squared_error'] )
    return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
EPOCHS = 1000;
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose = 0, callbacks=[early_stop, PrintDot()])

custom_data = pd.DataFrame({'conductivity': [3,5], 'water content': [4,7]})
print("shape ", custom_data.shape)
normed_custom_data = norm(custom_data)
custom_prediction = model.predict(normed_custom_data)
print ('prediction is ', custom_prediction)
dataframe_custom_prediction = pd.DataFrame(custom_prediction)
print(' dataframe custom prediction ', dataframe_custom_prediction)
dataframe_custom_prediction['conductivity'] = custom_data['conductivity']
dataframe_custom_prediction['water content'] = custom_data['water content']
answer_header = ['sand', 'clay', 'silt', 'conductivity', 'water_content']
dataframe_custom_prediction.to_csv('C:\\Users\\Computer\\Documents\\Example Soil Answers Data.csv', index=False, header=answer_header)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Training Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = "Validation Error")
    plt.ylim([0,10])
    plt.legend()

    plt.show()
    
plot_history(history)

loss, mean_absolute_error, mean_squared_error = model.evaluate(normed_test_data, test_labels, verbose = 0)
print("Testing set mean absolute error was ", mean_absolute_error)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('true values')
plt.ylabel('model predicted values')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]]) 
_ = plt.plot([-300,300], [-300,300])    
plt.show() 

print(test_predictions)
print('')
print(test_labels)
error = test_predictions - test_labels.values.flatten()
plt.hist(error, bins = 25)
plt.xlabel('prediction error')   
plt.ylabel('count')
plt.show()

