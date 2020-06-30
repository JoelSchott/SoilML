from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas
import seaborn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dry_250_location = "C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for GPR 250 dry 5-5-20.csv"
wet_250_location = "C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for GPR 250 wet 5-4-20.csv"
dry_500_location = "C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for GPR 500 dry 5-5-20.csv"
wet_500_location = "C:\\Users\\Computer\\Documents\\2020 Research\\Novelty data\\indices for GPR 500 wet 5-5-20.csv"

column_names = ["x (longitude)", "y", "cig", "cire", "gndvi", "ndre", "ndvi", "ndwi", "tcari", "vari", "VWC"]


def create_dataset(fileLocation):
    dataset = pandas.read_csv(fileLocation, names = column_names, skiprows = 1)
    return dataset.copy()

dataset = create_dataset(wet_500_location)
saved_data = dataset.copy()
description = dataset.describe()
print(description)


#seaborn.pairplot(dataset, x_vars = column_names, y_vars= ["VWC"])
#seaborn.pairplot(dataset, x_vars = ["VWC"], y_vars= ["VWC"], diag_kind = "kde")

#plt.show()

dataset.pop("x (longitude)")
dataset.pop("y")


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

def find_mae():
    training_data = dataset.sample(frac = 0.8, random_state = 0)
    test_data = dataset.drop(training_data.index)

    #for col in dataset.columns:
        #print(col)

    stats = training_data.describe()
    #print(stats)
    stats.pop("VWC")
    stats = stats.transpose()

    training_answers = training_data.pop("VWC")
    testing_answers = test_data.pop("VWC")

    training_data = norm(training_data, stats)
    test_data = norm(test_data, stats)

    model = build_model(len(training_data.keys()))
    #model.summary()

    EPOCHS = 1000

    early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)

    history = model.fit(training_data, training_answers, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

    #h = pandas.DataFrame(history.history)
    #h['epoch'] = history.epoch
    #print(h.tail())
    #print(h['mean_absolute_error'].iat[-1])

    #plot_history(history)

    loss, mean_absolute_error, mean_squared_error = model.evaluate(test_data, testing_answers, verbose = 0)
    #print ("this mean absolute error is ")
    #print (mean_absolute_error)
    return mean_absolute_error


for g in range (0, 8):
    index_name = column_names[g+2]
    index = saved_data.get(index_name)
    vwc = saved_data.get("VWC")
    frame = {index_name: index, "VWC": vwc}
    dataset = pandas.DataFrame(frame)
    mae = []
    for i in range (0,10):
        mae.append(find_mae())
    print("values are with only ", index_name)
    print(mae)
    print("average is ", average(mae))
    print("std is ", std_sample(mae))



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