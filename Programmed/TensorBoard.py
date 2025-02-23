import tensorflow
import datetime
import Tensor_Model
from tensorflow.keras import Model

#TODO: need to document the TensorModel and the IsolationModel

class TensorBoard(Tensor_Model, Model):
    def __init__(self):
        self._data = ""
        self._board = ""
        self._callback = ""
    
    def board_build():
        pass

    def board_compile():
        pass