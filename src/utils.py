import os
import sys
import dill 
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from scikeras.wrappers import KerasRegressor

from src.exception import CustomException
from src.logger import logging


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: Dict[str, Any], 
                    params: Dict[str, Any],
                    epochs: int = 10, batch_size: int = 32
                    
                    
                    ):
    """
    Evaluate multiple models using GridSearchCV for hyperparameter tuning and return their test R² scores.

    Parameters:
    - X_train (np.ndarray): Training data features
    - y_train (np.ndarray): Training data target
    - X_test (np.ndarray): Test data features
    - y_test (np.ndarray): Test data target
    - models (Dict[str, Any]): Dictionary of models to evaluate
    - params (Dict[str, Any]): Dictionary of hyperparameters for each model

    Returns:
    - report (Dict[str, float]): Dictionary of models with their test R² scores
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]


            if isinstance(model, KerasRegressor):
                para.update({'epochs': [epochs], 'batch_size': [batch_size]})

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score




            logging.info(f"{model} R² score on training data: {train_model_score}")
            logging.info(f"{model} R² score on test data: {test_model_score}")

          

        return report

    except Exception as e:
        raise CustomException(e, sys)






def convert_pickle_to_text(file_path):
    """
    Load a pickle file and convert its contents to a text representation.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    str: A text representation of the pickled content.
    """
    try:
        with open(file_path, 'rb') as file:
            content = pickle.load(file)
        
        # Convert the content to a string representation
        content_str = str(content)
        return content_str
    except FileNotFoundError:
        return f"The file {file_path} was not found."
    except Exception as e:
        return f"An error occurred while loading the pickle file: {e}"


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)




if  __name__=='__main__':
    pass
 
    preprocessor_pickle_file_path = "artifacts/preprocessor.pkl"
    model_pickle_file_path = "artifacts/model.pkl"
    pickled_content = convert_pickle_to_text(model_pickle_file_path)
    print(pickled_content)