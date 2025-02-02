import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):

    @abstractmethod
    def cal_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        pass

class MSE(Evaluation):

    def cal(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating Mean Square Error")
            mse= mean_squared_error(y_true,y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
    
class R2(Evaluation):

    def cal(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating R2 score")
            r2= r2_score(y_true,y_pred)
            logging.info(f"R2 score : {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e
        
class RMSE(Evaluation):

    def cal(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating Root Mean Square Error")
            rmse= root_mean_squared_error(y_true,y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
