from zenml import pipeline

from steps.getdata import get_file
from steps.cleandata import clean_file
from steps.evaluate import eval_model
from steps.modeltrain import train_model

@pipeline(enable_cache=True)
def training_pipeline(datapath: str):
    df=get_file(datapath)
    X_train, X_test, y_train, y_test =clean_file(df)
    model=train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse= eval_model(model,X_test, y_test)

     