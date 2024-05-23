import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')



class DataIngestion:
    def __init__(self):
       self.ingestin_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv('C:\\Workspace\\MlProjectGeneral\\notebook\\data\\stud.csv')
            logging.info("Read the dataset as datafreame")

            os.makedirs(os.path.dirname(self.ingestin_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestin_config.raw_data_path,index=False,header=True)
            logging.info("Saved raw data")

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestin_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestin_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestin_config.train_data_path,
                self.ingestin_config.test_data_path,
                #self.ingestin_config.raw_data_path
            )

        except Exception as e: 
            #pass
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        logging.info("Data transformation is completed")

    except Exception as e:
        logging.error("An error occurred during the data ingestion or transformation process", exc_info=True)




