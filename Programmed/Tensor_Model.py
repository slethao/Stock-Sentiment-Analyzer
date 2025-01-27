import tensorflow
from tensorflow.keras.layers import Dense, Dropout
import Isolation_Model as iso
from sklearn.preprocessing import StandardScaler
import pandas
import csv

class TensorModel():
    def __init__(self, group_chosen):
        self._group = group_chosen
        self._iso_obj = iso.IsolationModel(self._group)
        self._anomaly_scores = self._iso_obj.anomaly_results()
        self._data_used = self._iso_obj.get_data_used()

    def set_group(self, update_group):
        self._group = update_group

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
        # method 1
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        train_model.fit(features, self._iso_obj.get_x_value().values, epochs = 250, batch_size = 32,
               callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)],
               validation_split=0.424
              )  
        evaluate = train_model.evaluate(features, self._iso_obj.get_x_value().values)
        return evaluate

    def predict_model(self, train_model, file_path):
        # store predictions in dataframe and return the dataframe
        """
        negative value = predict has decrease for samples provided
            (predict lower value than base line) *from the baseline*
        positive value = predict has increase for samples provided
            (predicted higher value that predicted) *from the baseline*
        """
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values 
        scaler = StandardScaler() 
        features = scaler.fit_transform(features) 
        predict = train_model.predict(features)
        predict_data_frame = pandas.DataFrame({
                                                f"{self._group}": pandas.read_csv("Programmed/NVIDIA_STOCK_03.csv")[f"{self._group}"].values,
                                               f"Guess {self._group}": predict.flatten()}) # add the data set here
        predict_data_frame.to_csv(file_path, index = False)
        return predict_data_frame

#NOTE create a personal training method

# def main(): # create a driver file
#     #NOTE per column/attribute
#     file_path = "Programmed/NVIDIA_STOCK_PREDICT.csv"
#     ten = TensorModel("Adj Close")
#     ten.build_model()
#     model = ten.train_model()
#     print(ten.predict_model(model, file_path))
#     print("done done")

# main()