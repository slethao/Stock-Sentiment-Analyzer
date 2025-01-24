import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas
import numpy
import cleaning_data as clean

data = pandas.read_csv("Programmed/NVIDIA_STOCK_04.csv")
features = data.copy()
labels = features.pop('Adj Close')
features = numpy.array(features)
test_model = tensorflow.keras.Sequential([
                                            Dense(64, activation = 'relu'),
                                            Dense(1)
])
test_model.compile(loss = tensorflow.keras.losses.MeanSquaredError(),
                   optimizer = tensorflow.keras.optimizers.Adam())
test_model.fit(features, labels, epochs = 20) # minize loss but it works
noramalize = layers.Normalization()
noramalize.adapt(features)
norm_model = tensorflow.keras.Sequential([
                                            noramalize,
                                            layers.Dense(64, activation = 'relu'),
                                            layers.Dense(1)
])
norm_model.compile(loss = tensorflow.keras.losses.MeanSquaredError(),
                   optimizer = tensorflow.keras.optimizers.Adam())
norm_model.fit(features, labels, epochs = 20)
#NOTE reduce number of loss
#NOTE make sure its not overfit or under fit
#NOTE create a personal training method
#NOTE recreate the class


# class TensorModel(Model):
#     def __init__(self, filepath: str ,x_value, y_value, batch, learning_rate):
#         self._data = pandas.read_csv(filepath) # 1
#         self._x_value = numpy.array(x_value) # X_train
#         self._y_value = numpy.array(y_value) # y_train
#         self._batch = batch
#         self._learning_rate = learning_rate

#     def build_model(self): # 2 and 3
#         current_model = Sequential([ # need to watch out for overfitting
#                                     LSTM(units = 250, 
#                                             return_sequences=True, 
#                                             input_shape = (self._x_value.shape[1],1)),
#                                     Dropout(0.2),
#                                     LSTM(units = 250),
#                                     Dropout(0.2),
#                                     Dense(units = 1)
#         ])
#         return current_model

#     @tensorflow.function
#     def compile_model():
#         return TensorModel.build_model().compile(loss = 'mean_squared_error', optimizer = 'adam')

#     @tensorflow.function
#     def train_model(batch: tensorflow.data.Dataset, label: int): # 4
#         with tensorflow.GradientTape() as tape:
#             model = TensorModel.build_model()
#             predict = model(batch, training = True)
#             loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, predict)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         tensorflow.keras.optimizers.Adam().apply_gradients(zip(gradients, model.trainable_variables))
        
#         tensorflow.keras.metrics.Mean(name = 'train_loss')(loss)
#         tensorflow.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')(label, predict)
        
#         EPOCH = 20
#         for _ in range(EPOCH):
#             tensorflow.keras.metrics.Mean(name = 'train_loss').restart_state()
#             tensorflow.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy').reset_state()

#     @tensorflow.function
#     def conslusion(self):
#         """
#         Goal of this class: Find the daily returns and the price change
#         """
#         pass

# def main():
#     file_path = "Programmed/NVIDIA_STOCK_04.csv"
#     data_set = pandas.read_csv(file_path)
#     all_features = data_set.copy()
#     test_feature = all_features.pop("Adj Close")
#     all_features = numpy.array(all_features)
#     x_value = all_features  # x_train (all columns)
#     y_value = test_feature.values  # y_train (the last column)
#     learning_rate = 0.001
#     batch = 25
#     ten = TensorModel(file_path, x_value, y_value, batch, learning_rate) # through in values (first)
#     ten.build_model().fit(x_value, y_value, epochs=5) # build
#     glossary = clean.main()
#     ten.train_model(numpy.array(glossary.keys()), glossary['Adj Close']) # train (parameter) 
#     print("finsihed!!")
#     # conslusion

# main()