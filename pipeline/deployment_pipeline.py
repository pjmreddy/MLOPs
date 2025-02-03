import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


from steps.cleandata import clean_file
from steps.getdata import get_file
from steps.modeltrain import train_model
from steps.evaluate import eval_model


docker_settings= DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def cont_deploy_pipeline(
    datapath: str,
    min_accuracy: float =0.92,
    workers: int = 1,
    timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = get_file(datapath=datapath)
    X_train, X_test, y_train, y_test =clean_file(df)
    model=train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse= eval_model(model,X_test, y_test)
    mlflow_model_deployer_step(
        model = model,
        workers = workers,
        timeout = timeout,
    )

def inference_pipeline():
    pass