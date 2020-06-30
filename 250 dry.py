from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas
import seaborn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

column_names = ["x (longitude)", "y", "cig", "cire", "gndvi", "ndre", "ndvi", "ndwi", "tcari", "vari", "VWC (no zeros)"]

raw_dataset = pandas.read_csv("C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for GPR 250 dry 5-5-20.csv", names = column_names, skiprows=1)
dataset = raw_dataset.copy()

#seaborn.pairplot(dataset[column_names], diag_kind="kde")
seaborn.pairplot(dataset, x_vars = column_names, y_vars= ["VWC (no zeros)"])
plt.show()

dataset.pop("x (longitude)")
dataset.pop("y")

stats = dataset.describe()
print(stats)
stats.pop("VWC (no zeros)")
stats = stats.transpose()

answers = dataset.pop("VWC (no zeros)")

def norm(x):
    return (x - stats['mean']) / stats["std"]

dataset = norm(dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(dataset.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
    def _on_epoch_end(self, epoch, logs = None):
        if (epoch % 100 == 0):
            print('')
        print('.', end = '')

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 50)

history = model.fit(dataset, answers, epochs = EPOCHS, validation_split = 0.4, verbose = 0, callbacks = [early_stop, PrintDot()])

def plot_history(history):
    hist = pandas.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Mean absolute error")
    plt.plot(hist["epoch"], hist["mean_absolute_error"], label = "Train Error")
    plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label = "Test Error")
    plt.legend()

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("mean squared error")
    plt.plot(hist["epoch"], hist["mean_squared_error"], label = "train error")
    plt.plot(hist["epoch"], hist["val_mean_squared_error"], label = "test error")
    plt.legend()
    plt.show()

plot_history(history)