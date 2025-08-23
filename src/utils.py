import os # Para manejo de rutas y directorios
import sys   # Para acceder a información del sistema

import numpy as np
import pandas as pd
import  dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Importación de excepción
from src.exception import CustomException

# Función para guardar un objeto serializado
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Extrae el directorio desde la ruta del archivo

        os.makedirs(dir_path, exist_ok=True)  # Crea el directorio si no existe

        # Abre el archivo en modo binario de escritura ('wb')
        with open(file_path, 'wb') as file_obj:    
            dill.dump(obj, file_obj)  # Serializa el objeto y lo escribe en disco

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, # número de folds para validación cruzada.
                               #n_jobs=n_jobs, # número de núcleos
                               #verbose=verbose, # nivel de detalle en consola.
                               #refit=refit  # si True, reentrena el mejor modelo encontrado en todo el conjunto de entrenamiento
                               )
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            #model.fit(X_train, y_train) # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)