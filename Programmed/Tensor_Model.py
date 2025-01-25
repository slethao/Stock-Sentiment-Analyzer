import tensorflow
from tensorflow.keras.layers import Dense, Dropout
import Isolation_Model as iso
from sklearn.preprocessing import StandardScaler

isolation_model = iso.IsolationModel("Adj Close")
anomaly_scores = isolation_model.anomaly_results()

#ensure the dates are on there
data_used = isolation_model.get_data_used()
date_column = data_used['Date']
features = data_used.drop(columns=['Date', isolation_model.get_x_value().columns[0]]).values

scaler = StandardScaler()
features = scaler.fit_transform(features)

data_set = tensorflow.data.Dataset.from_tensor_slices((features, anomaly_scores, date_column))
data_set = data_set.shuffle(buffer_size = 1024).batch(32)

norm_model = tensorflow.keras.Sequential([ 
                                            tensorflow.keras.Input(shape=(features.shape[1],)),
                                            Dense(128, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                            Dropout(0.5),
                                            Dense(64, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                            Dropout(0.5),
                                            Dense(32, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                            Dropout(0.5),
                                            Dense(1) 
])

norm_model.compile(loss = tensorflow.keras.losses.MeanSquaredError(),
                   optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)) # learning rate is good

result = norm_model.fit(features, isolation_model.get_x_value().values, epochs = 250, batch_size = 32,
               callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)],
               validation_split=0.424
              ) 

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

# main()