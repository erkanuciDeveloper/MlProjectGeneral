# main purpose of data transformation:
#1 Feature engineering
#2 Data cleaning
#3 Data processing
#4 Categorical and Nume


import os
import sys
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# ColumnTransformer libray helpp to pipeline to encoding data
from sklearn.impute import SimpleImputer # for missing value we can use this
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # for save model we use this path
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function is responsible for creating a ColumnTransformer object for data transformation.
        '''
        try:
            # Define numeric and categorical columns
            numeric_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            


           # Define pipeline for numerical columns
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),  # Handle missing values
                ('scaler', StandardScaler())  # Scale the data
            ])


           # Define pipeline for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),  # Handle missing values
                ('one_hot_encoder', OneHotEncoder()),  # Convert categorical values to numerical using one-hot encoding
                ('scaler', StandardScaler(with_mean=False))  # Scale the data
            ])


            # Create a ColumnTransformer to apply the pipelines to the respective columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numeric_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )


            logging.info('Numerical columns standard scaler completed')
            logging.info('Categorical columns encoding completed')

            return preprocessor    

      

        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()


            # Define target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
     

            logging.info(f"Train DataFrame columns: {list(train_df.columns)}")
            logging.info(f"Test DataFrame columns: {list(test_df.columns)}")

    
            # Extract input and target features for train 
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Extract input and target features for  test datasets
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")


            # Apply preprocessing to train and test data
            # For trainin data we applyed 'fit_transform', and for test dataset we applyed 'transform'.
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)





            # This line of code is using NumPy to concatenate two arrays horizontally. 
            # Combine feature data and target labels into a single array fro training purposes.       
            # Concatenate input features and target labels
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]




           


            # Save preprocessing object to pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f'Saved preprocessing object')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e)
            
