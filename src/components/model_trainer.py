import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

# Ruta donde se guardará el objeto
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl') #ruta donde se guardará el modelo entrenado como archivo .pkl

# Clase para entrenar el modelo
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() # Inicializa la configuración 

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Dividiendo la data de entrada en train y test')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1], # toma todas las filas y todas las columnas excepto la última y guarda todo en X_train
                train_array[:,-1], # Última columna (target)
                test_array[:,:-1],
                test_array[:,-1]
                  
            )

            #Diccionario de modelos
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evalúa todos los modelos
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                             X_test=X_test, y_test=y_test, models=models)
            
            # Obtenemos el modelo con mejor score
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]  # Recupera el modelo correspondiente

            if best_model_score < 0.6:
                raise CustomException('No se encontro ningun modelo')
            
            logging.info(f'Se encontro el mejor modelo para train y test')

            # Guarda el modelo entrenado    
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)  # Realiza predicciones sobre test

            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)