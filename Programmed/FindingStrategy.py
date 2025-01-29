import pandas

class Strategies:
    def __init__(self, file_path):
        self._file_path = pandas.read_csv(file_path)
        self._predict_adjclose = ""
        self._predict_close = ""
    
    def recomendation(self):
        pass
