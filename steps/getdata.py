import logging

import pandas as pd # type: ignore
from zenml import step

class GetData:

    def __init__(self, datapath:str):
        self.datapath=datapath

    def collectdata(self):
        logging.info(f" Collecting data from {self.datapath}")
        return pd.read_csv(self.datapath)
    
@step
def get_file(datapath:str) -> pd.DataFrame:
    try:
        get_data= GetData(datapath)
        df=get_data.collectdata()
        return df
    except Exception as e :
        logging.error(f"Error while collecting data :{e}")
        raise e

    
