from sklearn.ensemble import IsolationForest
import pandas as pan


class IsolationModel:
    """
    This class is the blueprint for the object resposible 
    for creating an Isolation Forest Model.

    Attributes: 
        data_used = the content holds in the dataset
        x_value = the values that are held in a specific attribute
        model_obj = the obj to initilize the IsolationForest

    """
    def __init__(self, group: str):
        self._data_used = pan.read_csv("Programmed/Standard Filter/NVIDIA_STOCK_03.csv")
        self._x_value = self._data_used[[group]]
        self._model_obj = IsolationForest(contamination=0.1)
    

    """
    This method is used to display the entire given dataset.
    Args:
        None
    Returns:
        return the content of the entire dataset
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Isolation_Model.py as iso

            iso.IsolationModel(group).get_data_used()

        If written in the same file the method was invokded:
            get_data_used()
    """
    def get_data_used(self):
        return self._data_used


    """
    This method displays all the values within a attribute
    Args:
        None
    Returns:
        returns all the values within a attribute
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Isolation_Model.py as iso

            iso.IsolationModel(group).get_x_value()

        If written in the same file the method was invokded:
            get_x_value()
    """
    def get_x_value(self):
        return self._x_value


    """
    This method modifies whcih attribute the object is looking at
    Args:
        single_group = the updated value of the attribute
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Isolation_Model.py as iso

            iso.IsolationModel(group).set_x_value(single_group)

        If written in the same file the method was invokded:
            set_x_value(single_group)
    """
    def set_x_value(self, single_group):
        all_groups = self._data_used[["Adj Close","Close","High","Low","Open","Volume"]]
        if single_group in all_groups.columns:
            self._x_value = self._data_used[[single_group]] 
        else:
            print(f"The column ({single_group}) does not exisit in the csv file.")


    """
    This method finds the anomalies that are within the dataset
    Args:
        Nothing
    Returns:
        returns the anmoly score that incidates if values are an anmoly or not
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Isolation_Model.py as iso

            iso.IsolationModel(group).anomaly_results()

        If written in the same file the method was invokded:
            anomaly_results()
    """    
    def anomaly_results(self):
        self._model_obj.fit(self._x_value) # method
        anomaly_score = self._model_obj.decision_function(self._x_value) # method
        return anomaly_score
    

    """
    This method displays the outliers found in the dataset
    Args:
        Nothing
    Returns:
        returns the outliers of the dataset
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Isolation_Model.py as iso

            iso.IsolationModel(group).outlier_result()

        If written in the same file the method was invokded:
            outlier_result()
    """
    def outlier_result(self): 
        threshold = pan.Series(IsolationModel.anomaly_results(self)).quantile(0.05) # method
        outliers = self._data_used[IsolationModel.anomaly_results(self) < threshold][self._x_value.columns] # conclusion method
        return outliers


# def main():
#     iso = IsolationModel("Adj Close")
#     print(iso.outlier_result())
#     iso.set_x_value("Close")
#     print(iso.outlier_result())
#     print(iso.outlier_result())
#     print("done done")
# main()