"""
Regression!!
Example: fuel efficiency
"""

import matplotlib.pyplot
import numpy
import matplotlib
import pandas
import seaborn
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

numpy.set_printoptions(precision = 3, suppress = True)

# auto MPG dataset is downloaded here
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
group_names = ['MPG', 'Cylinders', 
               'Displacement', 'Horsepower', 
               'Weight', 'Acceleration',
               'Model Year', 'Origin']

raw_data = pandas.read_csv(url, names = group_names,
                           na_values = '?',
                           comment = '\t',
                           sep = ' ',
                           skipinitialspace=True)

print(raw_data)

# for this example just drop the values na (but this is not always the right choice when filtering out data)
raw_data = raw_data.dropna()

print(raw_data)

raw_data['Origin'] = raw_data['Origin'].map({1: 'USA',
                                             2: 'Europe',
                                             3: 'Japan'})

raw_data = pandas.get_dummies(raw_data, 
                              columns = ['Origin'], 
                              prefix = '',
                              prefix_sep = '')

print(raw_data.tail())

training_set = raw_data.sample(frac = 0.8, random_state=0)
test_set = raw_data.drop(training_set.index) #NOTE used as the final evaluation of the model

# inspect the data
seaborn.pairplot(training_set[['MPG', 
                               'Cylinders', 
                               'Displacement', 
                               'Weight']],
                               diag_kind = 'kde')
#NOTE stats on the like data count, mean, std, etc..
print(training_set.describe().transpose())

# split features from labels
train_features = training_set.copy()
test_features = test_set.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalization
training_set.describe().transpose()[['mean', 'std']]
norm = tensorflow.keras.layers.Normalization(axis = -1)
norm.adapt(numpy.array(train_features))
print(f"The mean and variance: {norm.mean.numpy()}")

# Linear regression
first = numpy.array(train_features[:1], dtype=float) # to ensure all the values are all the same type 'float'

with numpy.printoptions(precision=2, suppress=True):
    print(f"First example: {first}\n")
    print(f"Normalized: {norm(first).numpy()}")

# Regression with a deep neural netowrk (DNN)
horsepower = numpy.array(train_features['Horsepower'])
horsepower_norm = layers.Normalization(input_shape=[1,], axis = None)
horsepower_norm.adapt(horsepower)

# build model
horsepower_model = tensorflow.keras.Sequential([horsepower_norm,
                                                layers.Dense(units = 1)])
print(horsepower_model.summary())
# predict
print(horsepower_model.predict(horsepower[:10]))

# compile
horsepower_model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.1),
                         loss = 'mean_absolute_error')

# evaluate
execute_training = horsepower_model.fit(
                                        train_features['Horsepower'],
                                        train_labels,
                                        epochs = 100,
                                        verbose = 0,
                                        validation_split = 0.2
)

past = pandas.DataFrame(execute_training.history)
past['epoch'] = execute_training.epoch
print(past.tail())

#when showing the loss made in a method refer to lesson six (this is the same thing but in a method)
def plot_loss(history):
    matplotlib.pyplot.plot(execute_training.history['loss'], label = 'loss')
    matplotlib.pyplot.plot(execute_training.history['val_loss'], label = 'val_loss')
    matplotlib.pyplot.ylim([0, 10])
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Error [MPG]')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)

def plot_horsepower(x, y):
    matplotlib.pyplot.scatter(train_features['Horsepower'],
                              train_labels,
                              label = 'Data')
    matplotlib.pyplot.plot(x,
                           y,
                           color = 'k',
                           label = 'Predictions')
    matplotlib.pyplot.xlabel('Horsepower')
    matplotlib.pyplot.ylabel('MPG')
    matplotlib.pyplot.legend()

plot_loss(execute_training)
test_result = {}
test_result['horsepower_model'] = horsepower_model.evaluate(
                                                            test_features['Horsepower'],
                                                            test_labels,
                                                            verbose = 0
                                                            )

x_value = tensorflow.linspace(0.0, 250, 251)
y_value = horsepower_model.predict(x_value)
plot_horsepower(x_value, y_value)
# linear regression
linear_model = tensorflow.keras.Sequential([
                                            norm,
                                            layers.Dense(units=1)
                                            ])
linear_model.predict(train_features[:10])
linear_model.compile(
                      optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.1),
                      loss = 'mean_absolute_error'
                    )
perform = linear_model.fit(
                            train_features,
                            train_labels,
                            epochs=100,
                            verbose=0,
                            validation_split=0.2
                          )
plot_loss(execute_training)
test_result['linear_model'] = linear_model.evaluate(
                                                      test_features,
                                                      test_labels,
                                                      verbose=0
                                                    )

# Regression wtih deep neural network (DNN)
def build_and_compile_model(norm): #NOTE could be part of class that will build and compile the model
    model = keras.Sequential([
                                norm,
                                layers.Dense(64, activation = 'relu'),
                                layers.Dense(64, activation = 'relu'),
                                layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tensorflow.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(horsepower_norm)
dnn_model.summary()
train_hist = dnn_model.fit(train_features['Horsepower'],
                           train_labels,
                           validation_split = 0.2,
                           verbose = 0,
                           epochs = 100)
plot_loss(train_hist)
x = tensorflow.linspace(0.0, 250, 251)
y = dnn_model.predict(x)
plot_horsepower(x,y)
test_result['dnn_horsepower_model'] = dnn_model.evaluate(
                                                          test_features['Horsepower'],
                                                          test_labels,
                                                          verbose = 0)

# regression using a DNN and multiple inputs
multiple_dnn = build_and_compile_model(norm)
multiple_dnn.summary()

record = multiple_dnn.fit(
                           train_features,
                           train_labels,
                           validation_split =0.2,
                           verbose = 0,
                           epochs = 100)

plot_loss(record)
test_result['dnn_model'] = multiple_dnn.evaluate(test_features,
                                                 test_labels,
                                                 verbose = 0)

# Perforance
pandas.DataFrame(test_result, index = ['Mean absolute error [MPG]'])
test_predict = multiple_dnn.predict(test_features).flatten()
axis = matplotlib.pyplot.axes(aspect = 'equal')
matplotlib.pyplot.scatter(test_labels, test_predict)
matplotlib.pyplot.xlabel('True Values [MPG]')
matplotlib.pyplot.ylabel('Predictions [MPG]')
lims = [0, 50]
matplotlib.pyplot.xlim(lims)
matplotlib.pyplot.ylim(lims)
_ = matplotlib.pyplot.plot(lims, lims)
# error distribution
error = error = test_predict - test_labels
matplotlib.pyplot.hist(error, bins = 25)
matplotlib.pyplot.xlabel('Prediction Error [MPG]')
_ = matplotlib.pyplot.ylabel('Count')
#save model
multiple_dnn.save('dnn_model.keras')
# able to used the save model and relad it into the program
reload = tensorflow.keras.models.load_model('dnn_model.keras')
test_result['reloaded'] = reload.evaluate(
                                           test_features,
                                           test_labels,
                                           verbose = 0)
pandas.DataFrame(test_result, index = ['Mean absolute error [MPG]'])

# Conclusion


"""

"""