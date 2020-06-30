from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print (tf.__version__)

column_names = ['x', 'y', 'conductivity', 'water content', 'sand']
raw_dataset = pd.read_csv('C:\\Users\\Computer\\Documents\\Example Data.csv', names=column_names, skiprows=1)
dataset = raw_dataset.copy()
dataset.pop('x')
dataset.pop('y')

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

print("Train size ", train_dataset.size)
print("Test size ", test_dataset.size)

sns.pairplot(train_dataset[['conductivity', 'water content', 'sand']], diag_kind='kde')
plt.show()

train_stats = train_dataset.describe()
print(train_stats)
train_stats.pop('sand');
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('sand')
test_labels = test_dataset.pop('sand')

def norm(x):
    return (x - train_stats['mean'])  / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)
        ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()   
    
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')
        
EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=20)

history = model.fit(normed_train_data, train_labels, epochs= EPOCHS, validation_split=0.2, verbose = 0, callbacks=[early_stop, PrintDot()])

#data = {'conductivity': [3,4], 'water content': [12,3]}
custom_data = pd.DataFrame({'conductivity' : [3,4], 'water content' : [12,3]})
custom_data = norm(custom_data)
print('')
print(normed_test_data.shape)
print(normed_test_data)
print('')
print(custom_data.shape)
print(custom_data)
custom_prediction = model.predict(custom_data).flatten()
print('Prediction is ', custom_prediction)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('Mean Abs Error [Sand Units]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = "Validation Error")
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(hist['epoch'], hist['mean_squared_error'], label="Train Error")
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = "Validation Error")
    plt.ylim([0,20])
    plt.legend()
    plt.show()
    
plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("testing set mean absolute error was ", mae)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('true values')
plt.ylabel('model predicted values')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100,100],[-100,100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.show()