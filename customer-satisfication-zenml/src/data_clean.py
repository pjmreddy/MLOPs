import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self,df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            data = df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(),inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(),inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(),inplace=True)
            data["review_comment_message"].fillna("No_review",inplace=True)

            data=data.select_dtypes(include=[np.number])

            col=["customer_zip_code_prefix", "order_item_id"]
            data=data.drop(col,axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

class DivideStrategy(DataStrategy):
    
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
    
class DataCleaning:

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data=data
        self.strategy= strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in data handling: {e}")
            raise e
        