from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas
import seaborn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ec_location = "C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for EC dry.csv"
used_metric = "CH_1"
unused_metric = "CH_0.5"

column_names = ["x", "y", "cig", "cire", "gndvi", "ndre", "ndvi", "ndwi", "tcari", "vari", "CH_0.5", "CH_1"]

def create_dataset(fileLocation):
    dataset = pandas.read_csv(fileLocation, names = column_names, skiprows = 1)
    return dataset.copy()

dataset = create_dataset(ec_location)
saved_data = dataset.copy()
description = dataset.describe()
print(description)

#seaborn.pairplot(dataset[column_names], diag_kind="kde")

#seaborn.pairplot(dataset, x_vars = column_names, y_vars= [used_metric])
#seaborn.pairplot(dataset, x_vars = [used_metric], y_vars= [used_metric], diag_kind = "kde")

#plt.show()

dataset.pop("x")
dataset.pop("y")
dataset.pop(unused_metric)



def norm(x, stats):
    return (x - stats['mean']) / stats["std"]



def build_model(input_length):
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [input_length]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model



class PrintDot(keras.callbacks.Callback):
    def _on_epoch_end(self, epoch, logs = None):
        if (epoch % 100 == 0):
            print('')
        print('.', end = '')

def average(data):
    sum = 0;
    length = len(data)
    for i in range(0, length):
        sum += data[i]
    return sum / length

def std_sample(data):
    mean = average(data)
    sum = 0;
    length = len(data)
    for i in range(0, length):
        sum += (data[i] - mean)**2
    sum /= (length - 1)
    sum = sum**(1/2)
    return sum

def calculate_mae():
    training_data = dataset.sample(frac = 0.8, random_state = 0)
    test_data = dataset.drop(training_data.index)

    stats = training_data.describe()
    #print(stats)
    stats.pop(used_metric)
    stats = stats.transpose()

    training_answers = training_data.pop(used_metric)
    testing_answers = test_data.pop(used_metric)

    training_data = norm(training_data, stats)
    test_data = norm(test_data, stats)

    model = build_model(len(training_data.keys()))
    #model.summary()

    EPOCHS = 1000

    early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)

    history = model.fit(training_data, training_answers, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

    loss, mean_absolute_error, mean_squared_error = model.evaluate(test_data, testing_answers, verbose = 0)
    print("mean absolute error is ")
    print(mean_absolute_error)
    return mean_absolute_error

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

#plot_history(history)

for g in range (0, 8):
    index_name = column_names[g+2]
    index = saved_data.get(index_name)
    used = saved_data.get(used_metric)
    frame = {index_name: index, used_metric: used}
    dataset = pandas.DataFrame(frame)
    mae = []
    for i in range (0, 10):
        mae.append(calculate_mae())

    print("values are with only ", index_name)
    print(mae)
    print ("average is ", average(mae))
    print ("std is ", std_sample(mae))



#test_predictions = model.predict(test_data).flatten()

#plt.scatter(testing_answers, test_predictions)
#plt.xlabel("test answers")
#plt.ylabel("model prediction")
#plt.show()

#error = test_predictions - testing_answers
#plt.hist(error, bins = 25)
#plt.xlabel("prediction error")
#plt.ylabel("count")
#plt.show()