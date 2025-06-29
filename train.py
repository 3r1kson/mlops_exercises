import os
import random
import mlflow.keras
import numpy as np
import mlflow
import tensorflow as tf
import keras
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import tensorflow.keras as tf_keras


if not hasattr(tf_keras, "__version__"):
    tf_keras.__version__ = keras.__version__

# Configurações
SEED = 42
EPOCHS = 50
EXPERIMENT_NAME = "MLOps-Unit4-FetalHealth"
project_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos
model_save_path = os.path.join(project_dir, 'models', 'mlops_unit4.keras')
history_filepath = os.path.join(project_dir, 'mlopsCourse.pkl')

# MLOps config (tracking local, ou você pode setar URL do servidor remoto)
mlflow.set_tracking_uri("file://" + os.path.join(project_dir, "mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

# Semente para reprodutibilidade
def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

# Carrega e pré-processa os dados
data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"] - 1  # Classes 0, 1, 2

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = pd.DataFrame(scaler.fit_transform(X), columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=SEED
)

# Criação e compilação do modelo
reset_seeds()
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Experimento MLflow
with mlflow.start_run(run_name='unit4MLOps') as run:
    mlflow.log_param("epochs", EPOCHS)

    history = model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=3
    )

    # Salva histórico de treino
    with open(history_filepath, 'wb') as f:
        pickle.dump(history.history, f)
    mlflow.log_artifact(history_filepath)

    # Log de métricas por época
    for epoch in range(EPOCHS):
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

    # Inferência de assinatura
    input_example = X_train.iloc[:1].to_numpy()
    signature = infer_signature(X_train.to_numpy(), model.predict(X_train.to_numpy()))

    # Salva localmente
    model.save(model_save_path)

    # Loga no MLflow corretamente
    mlflow.keras.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )


    print(f"✅ Modelo salvo em: {model_save_path}")
    print(f"✅ Run ID: {run.info.run_id}")
