import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass #simplifica la creación de clases __init__, __repr__, etc.
# Un archivo .pkl es un archivo de objeto serializado creado con el módulo pickle de Python.\ 
# Es convertir un objeto (como un modelo, un DataFrame, una lista, etc.)\
#  en una secuencia de bytes que puede almacenarse en disco o transmitirse.

# Especifica la ruta donde se guardará el objeto de preprocesamiento (ColumnTransformer) serializado
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# clase principal que gestiona la transformación de datos
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Construye y devuelve ColumnTransformer que encapsula dos pipelines:
    def get_data_transformer_object(self):
        try:
            # Columnas numéricas a escalar
            numerical_columns = ['writing_score', 'reading_score']  
            # Columnas categóricas a codificar
            categorical_columns = ['gender',            
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course'
                                    ]
            
            # Pipeline para columnas numéricas: imputación + escalado:
            num_pipeline= Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]               
            )

            # Pipeline para columnas categóricas:    
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scarler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Columnas categóricas: {categorical_columns}')
            logging.info(f'Columnas numéricas: {numerical_columns}')

            # Combina ambos pipelines en ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipelines', cat_pipeline, categorical_columns)
                ]

            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys) # Eleva error con contexto del sistema

    # Aplica el objeto de transformación a los datos
    def initiative_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path) # cargar los conjuntos de train y test 
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            #Se invoca el método que construye el ColumnTransformer
            preprocessing_obj = self.get_data_transformer_object() 

            # Se separan las variables independientes de la variable objetivo
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            #Se aplica el preprocesamiento a los datos.
            logging.info(f'Aplicando preprocessing object en los conjuntos de train y test')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'preprocessing object guardado')
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

# np.c_ para concatenar arrays horizontalmente con numpy(similar a hacer con pandas pd.concat([col1, col2], axis=1))

        except Exception as e:
            raise CustomException(e,sys)
