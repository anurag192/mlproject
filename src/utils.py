import os
import sys
import numpy as np
import dill
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import Custom_Exception
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        print(e)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for name,model in models.items():
            
            model_params=params[name]
            gc=GridSearchCV(estimator=model,param_grid=model_params,cv=5,verbose=3,n_jobs=-1)
            gc.fit(X_train,y_train)
            model.set_params(**gc.best_params_)
            model.fit(X_train,y_train)
            

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[name]=test_model_score

        return report

    except Exception as e:
        raise Custom_Exception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)



