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



def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: Dict[str, Any], params: Dict[str, Any]):
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

        for model_name, model in models.items():
            logging.info(f"Training {model_name} with GridSearchCV")
            #param = params.get(model_name, {})

            # Perform Grid Search with cross-validation
            #gs = GridSearchCV(model, param, cv=3)
           # gs.fit(X_train, y_train)


           # Set best parameters and fit model
            #best_model = gs.best_estimator_
            #best_model.fit(X_train, y_train)  # Ensure model is fitted
            #logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

            # Predictions
            #y_train_pred = best_model.predict(X_train)
            #y_test_pred = best_model.predict(X_test)
            

            #fit the model, trainin model
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

       
            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score


            logging.info(f"{model_name} R² score on training data: {train_model_score}")
            logging.info(f"{model_name} R² score on test data: {test_model_score}")

          

        return report

    except Exception as e:
        raise CustomException(e, sys)






def convert_pickled_to_text(pickle_file_path):
    """
    Convert a pickled file to text representation.

    Args:
    - pickle_file_path (str): Path to the pickled file.

    Returns:
    - str: Text representation of the pickled content.
    """
    # Load the pickled object
    with open(pickle_file_path, "rb") as f:
        pickled_content = pickle.load(f)

    # Convert the pickled content to a human-readable format (text)
    text_representation = str(pickled_content)

    return text_representation


if  __name__=='__main__':
    pass
 
    preprocessor_pickle_file_path = "artifacts/preprocessor.pkl"
    model_pickle_file_path = "artifacts/model.pkl"
    text_representation = convert_pickled_to_text(model_pickle_file_path)
    print(text_representation)