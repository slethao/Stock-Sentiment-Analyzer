from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pan

class IsolationModel():
    def __init__(self):
        self._data_used = pan.read_csv("Programmed/NVIDIA_STOCK_04.csv")
        self._iso_forest = IsolationForest(contamination='auto', random_state=42)
        self._x_val = self._data_used[['Adj Close','Close','High','Low','Open','Volume']]
        self._y_val = self._data_used[['Adj Close']]
        # self._path_length =  self._iso_forest.get_path_lengths(self._x_val)
        # self._anomly_scores = self._iso_forest.decision_function(self._y_val)
    
    #NOTE need a setted for self._y_val

    def build_and_train(self):
        X_train, y_train = train_test_split(self._x_val, test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(self._x_val, self._y_val, test_size=0.2, random_state=42)
        self._iso_forest.fit(X_train)

    def decision_made(self): # prediction
        return self._iso_forest.predict(self._x_val)

    def result_found(self):
        return self._iso_forest.decision_function(self._x_val)

#@TODO: test below..
def main():
    iso_test = IsolationModel()
    iso_test.build_and_train()
    
    print(iso_test.decision_made())
    print(iso_test.result_found()[:10])
main()