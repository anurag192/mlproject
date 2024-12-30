import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','lunch','race_ethnicity','parental_level_of_education','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
        
            )
            logging.info("Numerical columns are standardized")
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder",OneHotEncoder()),
                    

                ]
            )
            logging.info("Categorical columns done")
            preprocessor=ColumnTransformer(
                [
                ("numerical pipeline",num_pipeline,numerical_columns),
                ("categorical pipeline",categorical_pipeline,categorical_columns)

                ]
               
            )
            return preprocessor
        
        except Exception as e:
            raise(Custom_Exception)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Train test completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_object()
            target_column_name="math_score"
          
            input_feature_train=train_df.drop(columns=target_column_name,axis=1)
            input_feature_test=test_df.drop(columns=target_column_name,axis=1)
            target_feature_train=train_df[target_column_name]
            target_feature_test=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test)
            ]
            logging.info("Saved processing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            print(e)






