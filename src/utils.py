import os
import sys
import numpy as np
import dill
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import Custom_Exception


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        print(e)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for name,model in models.items():
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[name]=test_model_score

        return report

    except Exception as e:
        raise Custom_Exception(e,sys)


