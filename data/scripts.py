# Get costume functions 

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from baseline.functions import smape,sample_fraction,load_data_clean,create_windows_from_sequences


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM
import torch
import csv
import numpy as np    
import tensorflow as tf
from tensorflow.keras import Model

def validate_data():
    test = os.path.isdir("Time-300B")
    if not test:
        print("Time-300B not found. Please download the dataset first using get_data.py script.")


def training_LSTM():
    """
    Train LSTM models 100% - 10% - 3% - 3% distilled
    """
    def build_lstm_model(window_size=15, n_features=1, lstm_units=50):
        model = Sequential([
            LSTM(lstm_units, input_shape=(window_size, n_features)),
            Dense(1)  # para predicción de un valor escalar siguiente
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    # Create directory for models if it doesn't exist
    os.mkdir('Models_lstm', exist_ok=True)
    
    ds = load_data_clean()

    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)
    
    # Traing for 100%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


    model = build_lstm_model(window_size=15, n_features=1, lstm_units=50)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[es]
    )

    # Calculate metrics
    y_pred = model.predict(X_val)
    y_pred_100 = y_pred.flatten()  

    mae = mean_absolute_error(y_val, y_pred_100)
    mse = mean_squared_error(y_val, y_pred_100)
    smape_value = smape(y_val, y_pred_100)

    print("Metrics for LSTM 100% Model:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)

    # Save the model
    model.save('Models_lstm/lstm_healthcare_model_100.keras')

    # Training for 10%

    X_10, y_10 = sample_fraction(X, y, 0.10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_10, y_10, test_size=0.2, shuffle=True, random_state=42)

    model = build_lstm_model(window_size=15, n_features=1, lstm_units=50)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[es]
    )
    # Calculate metrics
    y_pred = model.predict(X_val)
    y_pred_10 = y_pred.flatten()  

    print("Metrics for LSTM 10% Model:")
    mae = mean_absolute_error(y_val, y_pred_10)
    mse = mean_squared_error(y_val, y_pred_10)
    smape_value = smape(y_val, y_pred_10)

    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)

    # Guardar el modelo entrenado
    model.save('Models_lstm/lstm_healthcare_model_10.keras')


    # Training for 3%

    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)

    model = build_lstm_model(window_size=15, n_features=1, lstm_units=50)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[es]
    )

    # Calculate metrics
    y_pred = model.predict(X_val)
    y_pred_3 = y_pred.flatten()
    print("Metrics for LSTM 3% Model:")
    mae = mean_absolute_error(y_val, y_pred_3)
    mse = mean_squared_error(y_val, y_pred_3)
    smape_value = smape(y_val, y_pred_3)

    model.save('Models_lstm/lstm_healthcare_model_3.keras')

    # traing for 3% distilled

    X_3, y_3   = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)


    # validate teacher inference for not running it again if results already exist
    if not os.path.exists('teacher_results/val_teacher.csv') or not os.path.exists('teacher_results/train_teacher.csv'):
        print("Teacher results not found. Running teacher inference...")
    else:
        print("Teacher results found. Skipping teacher inference.")

    # Getting teacher inference
    # Teacher prediction using Time-MoE
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )

    train_predict_teacher = []
    val_predict_teacher = []

    for arr,y in zip(X_train,y_train):                   
        t = torch.from_numpy(arr.reshape(1, 15)).float()
        mean = t.mean(dim=-1, keepdim=True)   # (1,1)
        std  = t.std(dim=-1, keepdim=True)    # (1,1)
        std = std.clamp(min=1e-6) # Evitar división por cero
        normed_seq = (t - mean) / std         # (1,15)
        output = model.generate(normed_seq, max_new_tokens=1)  
        normed_pred = output[:, -1:]            # (1,1)
        pred = normed_pred * std + mean         # (1,1)
        train_predict_teacher.append(pred.item())  # Guardar las predicciones desnormalizadas

    for arr in X_val:                   
        t = torch.from_numpy(arr.reshape(1, 15)).float()
        mean = t.mean(dim=-1, keepdim=True)   # (1,1)
        std  = t.std(dim=-1, keepdim=True)    # (1,1)
        std = std.clamp(min=1e-6) # Evitar división por cero
        normed_seq = (t - mean) / std         # (1,15)
        output = model.generate(normed_seq, max_new_tokens=1)  
        normed_pred = output[:, -1:]            # (1,1)
        pred = normed_pred * std + mean         # (1,1)
        val_predict_teacher.append(pred.item())  # Guardar las predicciones desnormalizadas
    
    r_t_train =  np.array(train_predict_teacher).reshape(-1,1)
    r_t_val   =  np.array(val_predict_teacher).reshape(-1,1)
    t_train = np.array(y_train).reshape(-1,1)
    t_val   = np.array(y_val).reshape(-1,1)

    # Save teacher results for
    
    os.makedirs('teacher_results', exist_ok=True)

    with open('teacher_results/val_teacher.csv', mode='w', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(r_t_val)

    with open('teacher_results/train_teacher.csv', mode='w', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(r_t_train)


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
            self.tor_metric   = tf.keras.metrics.Mean(name="tor_loss")
            self.dist_metric  = tf.keras.metrics.Mean(name="distill_loss")
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
            outlier_loss = tf.sqrt(tf.abs(Rs - r_t_gt) + 1e-6)
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

            self.tor_metric.update_state(L_tor)
            self.dist_metric.update_state(L_dist)

            return {
                "loss": loss,
                "tor_loss": self.tor_metric.result(),
                "distill_loss": self.dist_metric.result(),
                "mse": self.metrics[0].result(),   # asumiendo que el primer self.metrics es MSE
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


    y_true_train = np.concatenate([t_train, r_t_train], axis=1)
    y_true_val   = np.concatenate([t_val,   r_t_val],   axis=1)


    xi    = t_train - r_t_train
    mad   = np.median(np.abs(xi - np.median(xi)))
    sigma = 1.4826 * mad

    alpha = 1.0

    ratio = alpha / (np.sqrt(2*np.pi) * sigma)
    ratio = min(ratio, 1 - 1e-6)

    epsilon = sigma * np.sqrt(-2.0 * np.log(ratio))

    student = StudentTOR(
        window_size=15,
        n_features=1,
        lstm_units=50,
        epsilon=epsilon,
        c_tor=1.0,
        c_dist=1.0
    )

    student.compile(optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])


    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = student.fit(
        X_train, y_true_train,
        validation_data=(X_val, y_true_val),
        epochs=20,
        batch_size=32,
        callbacks=[es]
    )


    Rs_output, Rd_output = student.predict(X_val)
    lstm_destillated = Rd_output.flatten() 

    # Metrics
    print("Metrics for LSTM Distillated Model:")

    mae = mean_absolute_error(y_val, lstm_destillated)
    mse = mean_squared_error(y_val, lstm_destillated)
    smape_val = smape(y_val, lstm_destillated)

    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_val)

    # Save the model
    student.save('Models_lstm/lstm_healthcare_model_distillated.keras')

    print("Training completed successfully. Models saved in 'Models_lstm' directory.")



def train_Neural():
    """ Train Neural Network models 100% - 10% - 3% - 3% distilled
    """
    return True