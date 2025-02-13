import logging
import mlflow
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple

from zenml.client import Client
experiment_tracker= Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval_model(model: RegressorMixin,
               X_test: pd.DataFrame,
               y_test: pd.DataFrame) -> Tuple[
                   
                   Annotated[float, "r2_score"],
                   Annotated[float, "rmse"]
               ]:
    try:
    
        prediction= model.predict(X_test)
        mse_class=MSE()
        mse= mse_class.cal_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)

        rmse_class=RMSE()
        rmse= rmse_class.cal_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        r2_class=R2()
        r2= r2_class.cal_scores(y_test,prediction)
        mlflow.log_metric("r2",r2)

        return  r2, rmse

    except Exception as e:
        logging.error(f"Error in Evaluating model: {e}")
        raise e
    