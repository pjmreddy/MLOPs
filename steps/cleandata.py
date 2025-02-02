import logging

from zenml import step
import pandas as pd
from src.data_clean import DataCleaning, DataPreprocessStrategy, DivideStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_file(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        procesStrategy = DataPreprocessStrategy()
        data_cleaning= DataCleaning(df, procesStrategy)
        processed_data= data_cleaning.handle_data()

        divideStrategy= DivideStrategy()
        data_cleaning= DataCleaning(processed_data, divideStrategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        return X_train, X_test, y_train, y_test
    except Exception as e: 
        logging.error(f"Error in cleaning data : {e}")
        raise e