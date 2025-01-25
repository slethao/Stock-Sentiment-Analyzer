from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pan


# data_used = pan.read_csv("Programmed/NVIDIA_STOCK_03.csv") # field
# x_values = data_used[["Adj Close","Close","High","Low","Open","Volume"]] # field
#test_model = IsolationForest(contamination=0.1) # field
# test_model.fit(x_values) # method
# anomaly_score = test_model.decision_function(x_values) # method
# threshold = pan.Series(anomaly_score).quantile(0.05) # method
# outliers = data_used[anomaly_score < threshold] # conclusion method
# print(outliers) # 76 datapoints are outliers

class IsolationModel:
    def __init__(self, group):
        self._data_used = pan.read_csv("Programmed/NVIDIA_STOCK_03.csv")
        self._x_value = self._data_used[[group]]
        # self._x_value = self._data_used[["Adj Close","Close","High","Low","Open","Volume"]]
        self._model_obj = IsolationForest(contamination=0.1)
    
    def set_x_value(self, single_group):
        all_groups = self._data_used[["Adj Close","Close","High","Low","Open","Volume"]]
        if single_group in all_groups.columns:
            self._x_value = self._data_used[[single_group]] 
        else:
            print(f"The column ({single_group}) does not exisit in the csv file.")
        
    def outlier_result(self): # fix this 
        self._model_obj.fit(self._x_value) # method
        anomaly_score = self._model_obj.decision_function(self._x_value) # method
        threshold = pan.Series(anomaly_score).quantile(0.05) # method
        outliers = self._data_used[anomaly_score < threshold][self._x_value.columns] # conclusion method
        return outliers


def main():
    iso = IsolationModel("Adj Close")
    print(iso.outlier_result())
    iso.set_x_value("Close")
    print(iso.outlier_result())
    iso.set_x_value("Stock")
    print(iso.outlier_result())
    print("done done")
main()