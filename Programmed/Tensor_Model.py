import tensorflow
from tensorflow.keras.layers import Dense, Dropout
import Isolation_Model as iso
from sklearn.preprocessing import StandardScaler
import pandas

class TensorModel():
    """
    The class is a blueprint of the object responsible 
    for create the TensorFlow machine learning model

    Attributes:
        group = the given group the user would like to see 
        iso_obj = the object that initilizes the Isolation Model (the clas made in Isolation_Model.py)
        anomaly_scores = the calculated scores on the values of the dataset to incide if they are anomalies
        data_used = the entire content of the datset chosen
    """
    def __init__(self, group_chosen:str):
        self._group = group_chosen
        self._iso_obj = iso.IsolationModel(self._group)
        self._anomaly_scores = self._iso_obj.anomaly_results()
        self._data_used = self._iso_obj.get_data_used()


    """
    This method is to modify what attribute will be seen.
    Args:
        update_group = updated chosen group
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Tensor_Model.py as ten

            ten.TensorModel(chosen_group).set_group(update_group)

        If written in the same file the method was invokded:
            set_group(update_group)
    """
    def set_group(self, update_group:str):
        self._group = update_group


    """
    This method is used to setup and build the machine learning model
    Args:
        Nothing
    Returns:
        returns the built TensorFlow model that is ready for use
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Tensor_Model.py as ten

            ten.TensorModel(chosen_group).build_model()

        If written in the same file the method was invokded:
            build_model()
    """
    def build_model(self):
        date_column = self._data_used['Date'] 
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values 
        scaler = StandardScaler() 
        features = scaler.fit_transform(features) 
        data_set = tensorflow.data.Dataset.from_tensor_slices((features, self._anomaly_scores, date_column)) 
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
        return norm_model


    """
    This model is used to train the machine learning model.
    Args:
        Nothing
    Returns:
        returns the trained TensorFlow model
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Tensor_Model.py as ten

            ten.TensorModel(chosen_group).train_model()

        If written in the same file the method was invokded:
            train_model()
    """
    def train_model(self):
        current_model = TensorModel(self._group).build_model()
        current_model.compile(loss = tensorflow.keras.losses.MeanSquaredError(),
                   optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)) # learning rate is good
        return current_model


    """
    This method is used to evaluate the machine learning model to compare the data loss versus data validation
    Args:
        train_model = the train model given
    Returns:
        returns the evaluated TensorFlow model
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Tensor_Model.py as ten

            ten.TensorModel(chosen_group).evaluate_model(train_model)

        If written in the same file the method was invokded:
            evaluate_model(train_model)
    """
    #@tensorflow.function
    def evaluate_model(self, train_model):
        features = self._data_used.drop(columns=['Date', self._iso_obj.get_x_value().columns[0]]).values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        train_model.fit(features, self._iso_obj.get_x_value().values, epochs = 250, batch_size = 32,
               callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)],
               validation_split=0.424
              )  
        evaluate = train_model.evaluate(features, self._iso_obj.get_x_value().values)
        return evaluate


    """
    This method is used to predict data for each attribute in the dataset 
    that was put into the machine learning model. This displays how accurate 
    how well the model can predict.
    Args:
        train_model = the train model given
        file_path = the location of where the dataset is stored
    Returns:
        returns the predicted data and actual data within a DataFrame
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Tensor_Model.py as ten

            ten.TensorModel(chosen_group).predict_model(train_model, file_path)

        If written in the same file the method was invokded:
            predict_model(train_model, file_path)
    """
    def predict_model(self, train_model, file_path):
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
        predict_data_frame = pandas.DataFrame({ "Date": pandas.read_csv("Programmed/Standard Filter/NVIDIA_STOCK_03.csv")["Date"].values,
                                                f"{self._group}": pandas.read_csv("Programmed/Standard Filter/NVIDIA_STOCK_03.csv")[f"{self._group}"].values,
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