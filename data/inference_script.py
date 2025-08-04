# Crear las series para las ciudades y crear las ventanas de cada una y despues generar un .csv con las metricas para graficar luego en el script de visualizacion

import pandas as pd
import numpy as np
import os 
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # silencia INFO y WARNING de TF C++
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # desactiva los mensajes de oneDNN

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from baseline.functions import smape,create_windows_inference
from baseline.functions import NeuralODEModel
os.makedirs('datos', exist_ok=True)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense

from transformers import AutoModelForCausalLM

def clean_data_covid(datos_covid: pd.DataFrame,dep:bool,name:str,save:bool=False):

    os.makedirs('datos', exist_ok=True)
    datos_covid.columns = (
        datos_covid.columns
        .str.strip()              # quitar espacios al inicio/fin
        .str.lower()              # pasar todo a minúsculas
        .str.replace(' ', '_')    # espacios → guiones bajos
    )
    # Estandarizo nombres de columna en minúsculas
    datos_covid = datos_covid.rename(columns=str.lower)
    # Ahora uso 'departamento' y 'fecha_inicio_sintomas'
    
    datos_covid['fecha_de_inicio'] = pd.to_datetime(
        datos_covid['fecha_de_inicio_de_síntomas'],
        errors='coerce'
    )
    datos_covid = datos_covid.dropna(subset=['fecha_de_inicio'])
    
    if dep:
        mask = datos_covid['nombre_departamento'].str.strip().str.upper() == name
        casos_santander = (
            datos_covid.loc[mask, 'fecha_de_inicio']
            .value_counts()
            .rename_axis('fecha')
            .reset_index(name='frecuencia')
            .sort_values('fecha')
        )
    else:
        mask = datos_covid['nombre_municipio'].str.strip().str.upper() == name
        casos_santander = (
            datos_covid.loc[mask, 'fecha_de_inicio']
            .value_counts()
            .rename_axis('fecha')
            .reset_index(name='frecuencia')
            .sort_values('fecha')
        )
    
    # 4) Rango completo de fechas
    fechas_completas = pd.date_range(
        start='2020-02-27',
        end='2024-01-15',
        freq='D'
    )

    df_sant = pd.DataFrame({'fecha': fechas_completas})
    df_sant = df_sant.merge(casos_santander, on='fecha', how='left')
    df_sant['frecuencia'] = df_sant['frecuencia'].fillna(0).astype(int)
    filename = os.path.join('datos', f"{name}.csv")
    if save:
        df_sant.to_csv(filename, index=False)

        print(f"Datos procesados y guardados en {filename}")
    return df_sant

# Load models
import torch
import tensorflow as tf

def load_neural_ode_models():
    # 1. Carga el checkpoint completo

    checkpoint = torch.load("data/Models_neural/modelo_3_dest.pth", map_location="cpu")
    checkpoint_2 = torch.load("data/Models_neural/modelo_3.pth", map_location="cpu")

    # 2. Filtra y renombra solo los pesos de 'backbone'
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("backbone.odefunc."):
            # elimino el prefijo "backbone."
            new_key = k.replace("backbone.", "")
            new_state_dict[new_key] = v


    ode_model = NeuralODEModel()    
    missing, unexpected = ode_model.load_state_dict(checkpoint, strict=False)
    ode_model.eval()


    new_state_dict = {}
    for k, v in checkpoint_2.items():
        if k.startswith("backbone.odefunc."):
            # elimino el prefijo "backbone."
            new_key = k.replace("backbone.", "")
            new_state_dict[new_key] = v

    ode_model_2 = NeuralODEModel()
    missing, unexpected = ode_model_2.load_state_dict(checkpoint_2, strict=False)
    ode_model_2.eval()

    return ode_model, ode_model_2


def load_LSTM_model():

    @tf.keras.utils.register_keras_serializable(package="CustomModels")
    class StudentTOR(Model):
        def __init__(
            self,
            window_size: int,
            n_features: int,
            lstm_units: int,
            epsilon: float,
            c_tor: float = 1.0,
            c_dist: float = 1.0,
            **kwargs
        ):
            super().__init__(**kwargs)
            # Guardamos los hiperparámetros para serializar
            self.window_size = window_size
            self.n_features  = n_features
            self.lstm_units  = lstm_units
            self.epsilon     = epsilon
            self.c_tor       = c_tor
            self.c_dist      = c_dist

            self.lstm = LSTM(
                lstm_units,
                input_shape=(window_size, n_features),
                name="lstm_layer"
            )
            self.dense_clean   = Dense(1, name="clean_output")
            self.dense_teacher = Dense(1, name="teacher_output")

        def call(self, inputs, training=False):
            x = self.lstm(inputs)
            return self.dense_clean(x), self.dense_teacher(x)

        def compute_tor_loss(self, t, r_t_gt, Rs):
            err = tf.abs(t - r_t_gt)
            clean_loss   = tf.square(Rs - t)
            outlier_loss = tf.sqrt(tf.square(Rs - r_t_gt) + 1e-6)
            return tf.where(err < self.epsilon, clean_loss, outlier_loss)

        def train_step(self, data):
            x, y = data
            t      = y[:, 0:1]   # etiqueta real
            r_t_gt = y[:, 1:2]   # pred teacher

            with tf.GradientTape() as tape:
                Rs, Rd = self(x, training=True)
                L_tor  = tf.reduce_mean(self.compute_tor_loss(t, r_t_gt, Rs))
                L_dist = tf.reduce_mean(tf.abs(Rd - r_t_gt))
                loss   = self.c_tor * L_tor + self.c_dist * L_dist

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.compiled_metrics.update_state(t, Rs)
            return {
                "loss": loss,
                "tor_loss": L_tor,
                "distill_loss": L_dist,
                **{m.name: m.result() for m in self.metrics}
            }

        def test_step(self, data):
            x, y = data
            t      = y[:, 0:1]
            r_t_gt = y[:, 1:2]
            Rs, Rd = self(x, training=False)

            L_tor  = tf.reduce_mean(self.compute_tor_loss(t, r_t_gt, Rs))
            L_dist = tf.reduce_mean(tf.abs(Rd - r_t_gt))
            loss   = self.c_tor * L_tor + self.c_dist * L_dist

            self.compiled_metrics.update_state(t, Rs)
            return {
                "loss": loss,
                "tor_loss": L_tor,
                "distill_loss": L_dist,
                **{m.name: m.result() for m in self.metrics}
            }

        def get_config(self):
            # Devuelve todo lo necesario para reconstruir la instancia
            base_config = super().get_config()
            return {
                **base_config,
                "window_size": self.window_size,
                "n_features":  self.n_features,
                "lstm_units":  self.lstm_units,
                "epsilon":     self.epsilon,
                "c_tor":       self.c_tor,
                "c_dist":      self.c_dist,
            }

        @classmethod
        def from_config(cls, config):
            # Separar kwargs de Model (si los hubiera)
            return cls(
                window_size=config.pop("window_size"),
                n_features=config.pop("n_features"),
                lstm_units=config.pop("lstm_units"),
                epsilon=config.pop("epsilon"),
                c_tor=config.pop("c_tor"),
                c_dist=config.pop("c_dist"),
                **config
            )

    lstm_model = tf.keras.models.load_model(
        "data/Models_lstm/lstm_healthcare_model_3_destilation.keras",
        compile=False
    )

    lstm_model_2 = tf.keras.models.load_model(
        "data/Models_lstm/lstm_healthcare_model_3.keras",
        compile=False
    )

    return lstm_model, lstm_model_2


def load_fundational_models():
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    return model

def main():

    # cities to analyze

    cities = {'CALI':False,
            'BOGOTA':False,
            'MEDELLIN':False,
            'SANTANDER':True,
            'ANTIOQUIA':True,
            'VALLE':True,
            'BOYACA':True,
            'CUNDINAMARCA':True}

    # Loading data
    data = pd.read_csv('data/covid_19.csv', low_memory=False)

    # Loading models
    neural_d , neural_n = load_neural_ode_models()
    lstm_d, lstm_n = load_LSTM_model()
    times_moe_model = load_fundational_models()

    results = []

    for city, dep in cities.items():    

        df = clean_data_covid(data, dep=dep, name=city,save=False)
        serie= df['frecuencia'].values
        X, y  = create_windows_inference(serie)

        # NeuralODE predictions
        x_neural = torch.tensor(X, dtype=torch.float32)
        pred_nd = neural_d(x_neural).cpu().detach().numpy().flatten()
        pred_nn = neural_n(x_neural).cpu().detach().numpy().flatten()

        # Metrics for NeuralODE
        smape_nd = smape(y, pred_nd)
        smape_nn = smape(y, pred_nn)
        mse_nd = mean_squared_error(y, pred_nd)
        mse_nn = mean_squared_error(y, pred_nn)
        mae_nd = mean_absolute_error(y, pred_nd)
        mae_nn = mean_absolute_error(y, pred_nn)


        # LSTM predictions
        _, Rd_output = lstm_d(X)
        pred_ld= Rd_output.cpu().numpy().flatten()
        
        output_lstm = lstm_n.predict(X)
        pred_ln = output_lstm.flatten() 

        # Metrics for LSTM
        smape_ld = smape(y, pred_ld)
        smape_ln = smape(y, pred_ln)
        mse_ld = mean_squared_error(y, pred_ld)
        mse_ln = mean_squared_error(y, pred_ln)
        mae_ld = mean_absolute_error(y, pred_ld)
        mae_ln = mean_absolute_error(y, pred_ln)


        # Times-MoE predictions

        # Pasa el modelo a modo evaluación
        times_moe_model.eval()

        train_predict_teacher = []

        # No gradientes ni cache
        with torch.no_grad():
            for arr in X:
                t = torch.from_numpy(arr.reshape(1, -1)).float()  
                mean = t.mean(dim=-1, keepdim=True)   # (1,1)
                std  = t.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (1,1)
                normed_seq = (t - mean) / std         # (1, window_size)

                outputs = times_moe_model(normed_seq)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits     
                else:
                    logits = outputs            
                if logits.ndim == 3:
                    normed_pred = logits[:, -1, :]   
                else:
                    normed_pred = logits[:, -1:]

                pred = (normed_pred * std + mean).item()
                train_predict_teacher.append(pred)


        # Metrics for Times-MoE
        pred_tmoe = np.array(train_predict_teacher)
        smape_tmoe = smape(y, pred_tmoe)
        mse_tmoe = mean_squared_error(y, pred_tmoe)
        mae_tmoe = mean_absolute_error(y, pred_tmoe)
    
        # add results
        results.append({
            'city': city,
            'type': 'NeuralODE_distilled',
            'SMAPE': smape_nd,
            'MSE': mse_nd,
            'MAE': mae_nd
        })
        results.append({
            'city': city,
            'type': 'NeuralODE_normal',
            'SMAPE': smape_nn,
            'MSE': mse_nn,
            'MAE': mae_nn
        })
        results.append({
            'city': city,
            'type': 'LSTM_distilled',
            'SMAPE': smape_ld,
            'MSE': mse_ld,
            'MAE': mae_ld
        })
        results.append({
            'city': city,
            'type': 'LSTM_normal',
            'SMAPE': smape_ln,
            'MSE': mse_ln,
            'MAE': mae_ln
        })
        results.append({
            'city': city,
            'type': 'Times-MoE',
            'SMAPE': smape_tmoe,
            'MSE': mse_tmoe,
            'MAE': mae_tmoe
        })
        print(f"Processed {city} - Results: {results[-3:]}")

    # Tras el bucle convertimos a DataFrame y volcamos a CSV
    df_results = pd.DataFrame(results)
    output_path = 'data/metrics.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Saved results in {output_path}")


if __name__ == '__main__':
    main()
