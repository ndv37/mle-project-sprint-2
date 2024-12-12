#собираю в 1 функцию

import click
import os
import psycopg
import pandas as pd
import mlflow
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dotenv import load_dotenv
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,mean_squared_error,r2_score

table_name="data_set_base_model"
load_dotenv()
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

dst_host = os.environ.get('DB_DESTINATION_HOST')
dst_port = os.environ.get('DB_DESTINATION_PORT')
dst_username = os.environ.get('DB_DESTINATION_USER')
dst_password = os.environ.get('DB_DESTINATION_PASSWORD')
dst_db = os.environ.get('DB_DESTINATION_NAME')
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
REGISTRY_MODEL_NAME ="MODEL_0"
    
dst_conn = create_engine(f'postgresql://{dst_username}:{dst_password}@{dst_host}:{dst_port}/{dst_db}')
data_set = pd.read_sql(f'select * from {table_name}', dst_conn)
EXPERIMENT_NAME = "SPRINT_2_PROJECT_EXP"
RUN_NAME = "SPRINT_2_base_model"

print(EXPERIMENT_NAME)
if mlflow.get_experiment_by_name(name=EXPERIMENT_NAME):
    experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']
    mlflow.set_experiment(experiment_id=experiment_id)
else:
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']
    mlflow.set_experiment(experiment_id=experiment_id)
    
#ОБУЧАЕМ НАШУ БАЗОВУЮ МОДЕЛЬ И ЛОГИРУЕМ
binary_cat_features=data_set[['has_elevator', 'is_apartment']]
num_features=data_set[['build_year', 'ceiling_height', 'floors_total', 'floor', 'rooms', 'total_area','flats_count']]
target=data_set['price']

#Объединяем трансформации
binary_cols = binary_cat_features.columns.tolist()
num_cols = num_features.columns.tolist()

# определите список трансформаций в рамках ColumnTransformer
preprocessor = ColumnTransformer([
    ('binary',OneHotEncoder(drop='if_binary'),binary_cols),
    ('num',StandardScaler(),num_cols)],verbose_feature_names_out=False,remainder='drop')

data_transformed_model_0 = preprocessor.fit_transform(data_set,data_set['price'])
data_transformed_model_0=pd.DataFrame(data_transformed_model_0, columns=preprocessor.get_feature_names_out())

data_transformed_model_0['price']=data_set['price']

x=data_transformed_model_0.drop(columns=['price'])
y=data_transformed_model_0['price']
X_tr, X_val, y_tr, y_val = train_test_split(x,y,test_size=0.2) 

#регрессия - предсказание числа
model=Ridge()
model.fit(X_tr,y_tr)
y_pred=model.predict(X_val)

features=pd.DataFrame(X_tr.columns)
features.columns=['feature']
features['coeff']=model.coef_/1000
features

df=pd.DataFrame(y_val/1000) 
df['y_pred']=y_pred/1000

signature = mlflow.models.infer_signature(X_val, y_val)
input_example = X_val[:10]

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    r2_sc = r2_score(y_val,y_pred)
    mse=mean_squared_error(y_val,y_pred)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models_sprint2",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        await_registration_for=60)
    mlflow.log_metric("r2", r2_sc)
    mlflow.log_metric("MSE", mse)
    #mlflow.log_artifact("artifacts/base_figure.png","artifacts")
