import tensorflow
import tf_keras
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas

class TensorModel(Model):
    def __init__(self, filepath: str ,x_value, y_value, batch, learning_rate):
        self._data = pandas.read_csv(filepath) # 1
        self._x_value = x_value # X_train
        self._y_value = y_value # y_train
        self._batch = batch
        self._learning_rate = learning_rate

    def build_model(self): # 2 and 3
        current_model = tf_keras.src.engine.sequential.Sequential([ # need to watch out for overfitting
                                                                    LSTM(units = 250, 
                                                                         return_sequences=True, 
                                                                         input_shape = (self._x_value.shape[1], 1)),
                                                                    Dropout(0.2),
                                                                    LSTM(units = 250),
                                                                    Dropout(0.2),
                                                                    Dense(units = 1)
        ])
        return current_model

    @tensorflow.function
    def compile_model():
        return TensorModel.build_model().compile(loss = 'mean_squared_error', optimizer = 'adam')

    @tensorflow.function
    def train_model(self, batch: tensorflow.data.Dataset, label: int): # 4
        with tensorflow.GradientTape() as tape:
            model = TensorModel.build_model()
            predict = model(batch, training = True)
            loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, predict)
        gradients = tape.gradient(loss, model.trainable_variables)
        tensorflow.keras.optimizers.Adam().apply_gradients(zip(gradients, model.trainable_variables))
        
        tensorflow.keras.metrics.Mean(name = 'train_loss')(loss)
        tensorflow.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')(label, predict)
        
        EPOCH = 20
        for _ in range(EPOCH):
            tensorflow.keras.metrics.Mean(name = 'train_loss').restart_state()
            tensorflow.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy').reset_state()

    @tensorflow.function
    def conslusion(self):
        """
        Goal of this class: Find the daily returns and the price change
        """
        pass

def main():
    file_path = "Programmed/NVIDIA_STOCK_04.csv"
    x_value = 0 # x_train (all columns)
    y_value = 0 # y_train (the last column)
    learning_rate = 0.001
    batch = 256
    ten = TensorModel(file_path, x_value, y_value, batch, learning_rate) # through in values (first)
    ten.build_model() # build
    ten.train_model() # train
    print("finsihed!!")
    # conslusion

main()