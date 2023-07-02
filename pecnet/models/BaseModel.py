import logging
import os

import tensorflow.keras as tf_kr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

logger = logging.getLogger(__name__)

class BaseModel():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf_kr.Sequential()

    def get_model(self):
        return self.model

    def get_model_info(self):
        raise NotImplementedError
    
    #Set model.fit args in child.
    def set_fit_args(self):
        raise NotImplementedError

    #This is for logging
    def set_model_info(self):
        raise NotImplementedError

    #Define model here in child classes
    def init_model(self):
        raise NotImplementedError

    # def is_fit_model_implemented(self):
    #     return False

    #Fits the model. It is for overriding input output shape etc.
    def fit_model(self):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)