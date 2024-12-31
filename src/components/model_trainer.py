import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as calculated_r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object
from sklearn.model_selection import train_test_split
from src.utils import evaluate_model
from sklearn.model_selection import GridSearchCV
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Train test split initialised")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],

            )
            models={
                "RandomForestRegression":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "Xgboost":XGBRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }
            params={
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_depth':[5,8,10,15],
                    'max_features':['auto','sqrt','log2']
                },
                "RandomForestRegression":{
                    'criterion':['squared_error','absolute_error','friedman_mse'],
                    'n_estimators':[8,18,32,64,128,256],
                    'max_features':['auto','sqrt','log2']
                    
                },
                "Gradient Boosting":{
                    'loss':['squared_error','absolute_error'],
                    'n_estimators':[8,16,32,64,128],
                    'max_depth':[5,8,10,15],
                    'max_features':['auto','sqrt','log2']

                },
                "Linear Regression":{},
                "Xgboost":{
                    'learning_rate':[0.01,0.1,0.5],
                    'n_estimators':[8,16,32,64]
                },
                "Adaboost Regressor":{
                    'learning_rate':[0.01,0.1,0.5,0.01],
                    'n_estimators':[8,16,32,64,128]
                },
                "KNeighborsRegressor":{
                    'n_neighbors':[5,8,10,15],
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree']

                }
            }
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<=0.6:
                raise Custom_Exception("No model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("R2 score will be calculated")
            predicted=best_model.predict(X_test)
            test_r2_score=calculated_r2_score(y_test,predicted)
            logging.info(f"R2 score will be {test_r2_score}")


            return test_r2_score
    

        except Exception as e:
            raise Custom_Exception(e,sys)

