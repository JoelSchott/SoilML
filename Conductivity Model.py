# this a modified version of an example found at https://www.tensorflow.org/tutorials/keras/regression

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas
import seaborn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# data that will be used, eventually will consist of conductivity readings and various visual indices
# we are trying to use some combination of the indices to predict conductivity
# in this example "z" refers to an index used to predict conductivity
column_names = ["z average", "z sd", "conductivity"]
raw_dataset = pandas.read_csv("C:\\Users\\Computer\\Documents\\ML_ready_data.csv", names = column_names, skiprows = 1)

dataset = raw_dataset.copy()

# using 80 percent of the data for training and the rest for testing
training_data = dataset.sample(frac = 0.8, random_state = 0)
test_data = dataset.drop(training_data.index)

# percentage of 80% is somewhat arbitrary and can be changed,
# maybe to have training data be everything and test data to be a random 20 percent

# training_data = dataset
# test_data = dataset.sample(frac = 0.2, random_state = 0)

# although that would would mean that the test data is a sub-group of the training data, which maybe should be avoided


print("Training data size are ", training_data.size)
print("testing data size are ", test_data.size)

# a visual comparison helpful for seeing trends
seaborn.pairplot(dataset[column_names], diag_kind="kde")
plt.show()

# a look at general statistics, transposed so mean and standard deviation are columns
train_stats = training_data.describe()
print(train_stats)
train_stats.pop("conductivity")
train_stats = train_stats.transpose()

# creates the targets and removes the answers from the training data
train_answers = training_data.pop("conductivity")
test_answers = test_data.pop("conductivity")

# normalizing accounts for the potential different scales and ranges of the indices
# this may not be strictly necessary, but it is makes training easier
# also the units of the indices are not important as long as they are consistent
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

# the test data is normed by the training data in order to make the test data relative to the training data that
# the model has been trained on
training_data = norm(training_data)
test_data = norm(test_data)

# the model will have two hidden layers of 64 nodes each, and a final layer of one that will be the predicted conductivity
# this is a bit of a black box, I don't know what changes to the model would make it more effective
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape  = [len(training_data.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()

# prints a '.' for every time the model runs and changes values, going to a new line every 100 iterations
class PrintDot(keras.callbacks.Callback):
    def _on_epoch_end(self, epoch, logs = None):
        if (epoch % 100 == 0):
            print('')
        print('.', end = '')

EPOCHS = 1000
# patience is how many times to run, without improvement, before stopping
# if the model is run too many times, it begins to "memorize" the training data and becomes less effective on testing data
# this is called overfitting, and both early stopping and a network with few hidden layers can prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)

# making the model, validation split refers to how much of the training data is used to evaluate the training
history = model.fit(training_data, train_answers, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

# making a graph of the training
def plot_history(history):
    hist = pandas.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Mean absolute error")
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Testing Error')
    plt.legend()

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = "train error")
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'test error')
    plt.legend()
    plt.show()

plot_history(history)

# measures of how well the model evaluated the test data (the random 20% at the beginning)
loss, mean_absolute_error, mean_squared_error = model.evaluate(test_data, test_answers, verbose = 0)

# data that are the predictions of the model
test_predictions = model.predict(test_data).flatten()

# visual representation of the predictions
plt.scatter(test_answers, test_predictions)
plt.xlabel("Test Data")
plt.ylabel("Model Predicted Values")
plt.show()

# visual representation of the error
error = test_predictions - test_answers
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()

#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#