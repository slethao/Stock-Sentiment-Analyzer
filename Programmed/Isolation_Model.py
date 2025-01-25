from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pan


class IsolationModel:
    def __init__(self, group):
        self._data_used = pan.read_csv("Programmed/NVIDIA_STOCK_03.csv")
        self._x_value = self._data_used[[group]]
        self._model_obj = IsolationForest(contamination=0.1)
    
    def get_data_used(self):
        return self._data_used

    def get_x_value(self):
        return self._x_value

    def set_x_value(self, single_group):
        all_groups = self._data_used[["Adj Close","Close","High","Low","Open","Volume"]]
        if single_group in all_groups.columns:
            self._x_value = self._data_used[[single_group]] 
        else:
            print(f"The column ({single_group}) does not exisit in the csv file.")
        
    def anomaly_results(self):
        self._model_obj.fit(self._x_value) # method
        anomaly_score = self._model_obj.decision_function(self._x_value) # method
        return anomaly_score
    
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