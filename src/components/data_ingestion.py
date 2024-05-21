import os
import sys
from  src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
#we will siplit the data into this folder path
class DataIngectionConfig:
    # Paths to save the split data
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    row_data_path: str=os.path.join('artifacts','row.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngectionConfig() #when we execute this code above 3 particular class variable comes inside this particular

    
    def initation_data_ingestion(self):
        logging.info('Entered the data ingestion method or componet')

        try:
                  
            # Read the dataset into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')   
            logging.info('Read the dataset as dataframe')



            # cretate path for split above type data. Train, Test and Row Data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True) 
            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)


            # Save the raw dataset
            df.to_csv(self.ingestion_config.row_data_path, index=False, header=True)
            logging.info('Saved the raw dataset')

            logging.info('Train-test split initiated')

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)         
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is completed')


            # Return the paths to the train and test data
            # We need to train and test data path only , data why in retun we writed just two paramaters.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:

            raise CustomException(e,sys) 
        


# Now we can run this file for creating ,train,test and row data to into articacts folder.
if __name__=='__main__':

    obj=DataIngestion()
    obj.initation_data_ingestion()
            

    


