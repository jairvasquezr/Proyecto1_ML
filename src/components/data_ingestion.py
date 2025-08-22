import os 
import sys
from src.exception import CustomException # Manejo de errores
from src.logger import logging # Logger centralizado para trazabilidad

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Rutas para el guardado
@dataclass
class DataIngestionConfig:
    train_data_path = str=os.path.join('artifacts','train.csv')
    test_data_path = str=os.path.join('artifacts','test.csv')
    raw_data_path = str=os.path.join('artifacts','data.csv')


# Clase principal para la ingestión de datos
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Ingresa el método de data ingestion')
        try:
            logging.info('Carga del dataset como dataframe')
            df=pd.read_csv('notebook\data.csv') # Fuente de datos (puede ser archivo, API, DB, etc.)

            # Crea carpeta destino si no existe
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Guarda el dataset original como respaldo
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Inicio Train test split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=123)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingesta de datos se a completado')
             # Devuelve rutas para uso posterior en el pipeline
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation  = DataTransformation()
    data_transformation.initiative_data_transformation(train_data, test_data)
