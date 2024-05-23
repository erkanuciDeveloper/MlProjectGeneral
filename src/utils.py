import os
import sys
import dill 

import numpy as np
import pandas as pd

import pickle

from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)





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
 
    #pickle_file_path = "artifacts/preprocessor.pkl"
    #text_representation = convert_pickled_to_text(pickle_file_path)
    #print(text_representation)