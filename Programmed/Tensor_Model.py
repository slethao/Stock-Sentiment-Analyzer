import tensorflow
from tensorflow.keras.layers import Dense, Dropout
import Isolation_Model as iso
from sklearn.preprocessing import StandardScaler

class TensorModel():
    def __init__(self, group_chosen):
        self._iso_obj = iso.IsolationModel(group_chosen)
        self._anomaly_scores = self._iso_obj.anomaly_results()
        self._data_used = self._iso_obj.get_data_used()
        self._group = group_chosen

    def build_model(self):
        date_column = self._data_used['Date'] # method 1
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values # method 1
        scaler = StandardScaler() # method 1
        features = scaler.fit_transform(features) # method 1
        data_set = tensorflow.data.Dataset.from_tensor_slices((features, self._anomaly_scores, date_column)) # method 1
        data_set = data_set.shuffle(buffer_size = 1024).batch(32) # method 1

        norm_model = tensorflow.keras.Sequential([ 
                                                    tensorflow.keras.Input(shape=(features.shape[1],)),
                                                    Dense(128, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                                    Dropout(0.5),
                                                    Dense(64, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                                    Dropout(0.5),
                                                    Dense(32, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)),
                                                    Dropout(0.5),
                                                    Dense(1) 
        ]) # method 1
        return norm_model

    def train_model(self):
        current_model = TensorModel(self._group).build_model()
        current_model.compile(loss = tensorflow.keras.losses.MeanSquaredError(), # method 2
                   optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)) # learning rate is good method 2
        return current_model

    #@tensorflow.function
    def evaluate_model(self, train_model):
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values # method 1
        scaler = StandardScaler() # method 1
        features = scaler.fit_transform(features) # method 1
        result = train_model.fit(features, self._iso_obj.get_x_value().values, epochs = 250, batch_size = 32,
               callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)],
               validation_split=0.424
              )  # method 3
        return result

    @tensorflow.function
    def conslusion(self):
        """
        Goal of this class: Find the daily returns and the price change
        """
        pass

#NOTE create a personal training method

def main():
    ten = TensorModel("Adj Close")
    ten.build_model()
    model = ten.train_model()
    ten.evaluate_model(model)

main()