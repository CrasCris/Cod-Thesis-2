# Get costume functions 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # 0 = todos, 1 = INFO y superior, 2 = WARNING+, 3 = ERROR+

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

import torch.optim as optim
from torchdiffeq import odeint
import torch
import torch.nn as nn

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
    else:
        print("Teacher results found. Skipping teacher inference.")
        # Leer validación
        r_t_val = []
        with open('teacher_results/val_teacher.csv', newline='') as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
                fila_convertida = [float(x) for x in fila]
                r_t_val.append(fila_convertida)

        # Leer entrenamiento
        r_t_train = []
        with open('teacher_results/train_teacher.csv', newline='') as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                fila_convertida = [float(x) for x in fila]
                r_t_train.append(fila_convertida)


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


    # --- calcular xi, mad, sigma (ya lo tenías casi bien) ---
    xi = t_train - r_t_train
    xi = xi.reshape(-1)  # asegurar shape 1D
    mad = np.median(np.abs(xi - np.median(xi)))
    sigma = 1.4826 * mad

    alpha = 1.0
    B = float(len(xi))  # sample number (paper uses B)

    # inner_term according to Eq.(3): sqrt(2*pi)*sigma*alpha / B
    inner = np.sqrt(2.0 * np.pi) * sigma * alpha / (B + 1e-12)

    # clamp inner to (tiny, <1) to avoid log invalid or negative sqrt
    inner = np.clip(inner, 1e-12, 1.0 - 1e-12)

    epsilon = sigma * np.sqrt(-2.0 * np.log(inner))

    # epsilon ready
    print("sigma", sigma, "epsilon", epsilon)


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

    os.makedirs('Models_neural', exist_ok=True)
    # Define the neural network model
    # Neural ODE Model
    class ODEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 50),
                nn.Tanh(),
                nn.Linear(50, 1)
            )

        def forward(self, t, x):
            return self.net(x)

    class NeuralODEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.odefunc = ODEFunc()
        
        def forward(self, x):
            # x: [batch_size, seq_len, features]
            device = x.device
            seq_len = x.shape[1]
            # Crear t en el dispositivo correcto
            t = torch.linspace(0, 1, seq_len, device=device)
            # Estado inicial para cada muestra: último valor de la secuencia
            # Si features=1, x[:, -1, :] es [batch_size, 1]
            y0 = x[:, -1, :]
            # Integrar en batch: devuelve [len(t), batch_size, features]
            out = odeint(self.odefunc, y0, t, method='rk4')
            # Tomar el valor final en t=1 para cada muestra
            y_final = out[-1]
            return y_final  # forma [batch_size, features]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    def train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_vals = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            loss_vals.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        best_epoch = int(np.argmin(loss_vals)) + 1
        best_loss = loss_vals[best_epoch - 1]
        print(f"Mejor época = {best_epoch}, Loss = {best_loss:.4f}")

        # Evaluación
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
            y_pred = model(X_test_tensor).cpu().numpy()

        y_test_inv = y_test_tensor.cpu().numpy().reshape(-1, 1)
        mae = mean_absolute_error(y_test_inv, y_pred)
        mse = mean_squared_error(y_test_inv, y_pred)
        smape_value = smape(y_test_inv, y_pred)
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, sMAPE: {smape_value:.4f}")

        return model, loss_vals
    
    ds = load_data_clean()
    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)

    # Training for 100% 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = NeuralODEModel()
    trained_model, loss_vals = train_model(model, X_train, y_train, X_test, y_test)

    # metrics
    y_pred = trained_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)

    print("Metrics for LSTM 100% Model:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)
    
    torch.save(trained_model.state_dict(), "Models_neural/modelo_100.pth")

    # Training for 10%
    X_10, y_10 = sample_fraction(X, y, 0.10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_10, y_10, test_size=0.2, shuffle=True, random_state=42)
    model = NeuralODEModel()
    trained_model, loss_vals = train_model(model, X_train, y_train, X_test, y_test)

    # metrics
    y_pred = trained_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)

    print("Metrics for LSTM 10% Model:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)
    
    torch.save(trained_model.state_dict(), "Models_neural/modelo_10.pth")

    # Training for 3%   
    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)
    model = NeuralODEModel()
    trained_model, loss_vals = train_model(model, X_train, y_train, X_test, y_test)

    # metrics
    y_pred = trained_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)

    print("Metrics for LSTM 3% Model:") 
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)

    torch.save(trained_model.state_dict(), "Models_neural/modelo_3.pth")
    # Training for 3% distilled
    X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)


    # Load teacher predictions
    import csv

    # Leer validación
    r_t_val = []
    with open('teacher_results/val_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
            fila_convertida = [float(x) for x in fila]
            r_t_val.append(fila_convertida)

    # Leer entrenamiento
    r_t_train = []
    with open('teacher_results/train_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            fila_convertida = [float(x) for x in fila]
            r_t_train.append(fila_convertida)

    class StudentODE(nn.Module):
        def __init__(self):
            super().__init__()
            # tu ODE backbone
            self.backbone = NeuralODEModel()
            # dos salidas
            self.head_clean   = nn.Linear(1, 1)    # Rs
            self.head_distill = nn.Linear(1, 1)    # Rd

        def forward(self, x):
            # x: [batch, seq_len, feat=1]
            h = self.backbone(x)                   # [batch,1]
            Rs = self.head_clean(h)                # [batch,1]
            Rd = self.head_distill(h)              # [batch,1]
            return Rs, Rd
    
    def train_student_ode(
        model, X_train, y_train, r_t_train,
        X_val,   y_val,   r_t_val,
        batch_size=64, epochs=20, lr=0.001
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        # DataLoaders que entregan (Xb, tb, rtb)
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1,1), dtype=torch.float32),
            torch.tensor(r_t_train,    dtype=torch.float32)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val.reshape(-1,1), dtype=torch.float32),
            torch.tensor(r_t_val,    dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Pre‑calcula epsilon con clamp para no obtener nan
        # 1) Calcula xi
        xi  = y_train.reshape(-1,1) - r_t_train
        xi = xi.reshape(-1)  # asegurar shape 1D
        mad = np.median(np.abs(xi - np.median(xi)))
        sigma = 1.4826 * mad

        alpha = 1.0
        B = float(len(xi))  # sample number (paper uses B)

        # inner_term according to Eq.(3): sqrt(2*pi)*sigma*alpha / B
        inner = np.sqrt(2.0 * np.pi) * sigma * alpha / (B + 1e-12)

        # clamp inner to (tiny, <1) to avoid log invalid or negative sqrt
        inner = np.clip(inner, 1e-12, 1.0 - 1e-12)

        epsilon = sigma * np.sqrt(-2.0 * np.log(inner))

        # epsilon ready
        print("sigma", sigma, "epsilon", epsilon)

        for epoch in range(1, epochs+1):
            model.train()
            tot_loss = 0.0

            for Xb, tb, rtb in train_loader:
                Xb, tb, rtb = Xb.to(device), tb.to(device), rtb.to(device)
                optimizer.zero_grad()

                Rs, Rd = model(Xb)  # ambas de shape (B,1)

                # TOR loss:
                err      = torch.abs(tb - rtb)
                loss_c   = (Rs - tb).pow(2)
                loss_o   = torch.sqrt((Rs - rtb).pow(2) + 1e-6)
                L_tor    = torch.where(err < epsilon, loss_c, loss_o).mean()

                # Distillation loss (L1):
                L_dist   = torch.abs(Rd - rtb).mean()

                loss = L_tor + L_dist
                loss.backward()
                optimizer.step()
                tot_loss += loss.item() * Xb.size(0)



            tot_loss /= len(train_ds)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {tot_loss:.4f}")
        return model

    student = StudentODE()

    student_trained = train_student_ode(
        student,
        X_train, y_train, r_t_train,
        X_test,   y_test,   r_t_val,
        batch_size=64,
        epochs=20,
        lr=0.001
    )

    student_trained.eval()
    Rs_pred, Rd_pred = student_trained(torch.tensor(X_test, dtype=torch.float32).to(device))
    # Rd_pred es tu salida distill, compárala con y_val:
    results =  Rd_pred.cpu().detach().numpy().flatten()

    # metrics

    mae = mean_absolute_error(y_test, results)
    mse = mean_squared_error(y_test, results)
    smape_value = smape(y_test, results)

    print("Metrics for LSTM 3% Model:") 
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)
    torch.save(student.state_dict(), "Models_neural/modelo_3_dest.pth")

    return True


def train_Neural_KL():
    """Train Neural Network 3% distilled using a new Loss function based on KL Divergence"""

    os.makedirs('Models_neural', exist_ok=True)

    # Neural ODE Model
    class ODEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 50),
                nn.Tanh(),
                nn.Linear(50, 1)
            )

        def forward(self, t, x):
            return self.net(x)

    class NeuralODEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.odefunc = ODEFunc()
        
        def forward(self, x):
            # x: [batch_size, seq_len, features]
            device = x.device
            seq_len = x.shape[1]
            # Crear t en el dispositivo correcto
            t = torch.linspace(0, 1, seq_len, device=device)
            # Estado inicial para cada muestra: último valor de la secuencia
            # Si features=1, x[:, -1, :] es [batch_size, 1]
            y0 = x[:, -1, :]
            # Integrar en batch: devuelve [len(t), batch_size, features]
            out = odeint(self.odefunc, y0, t, method='rk4')
            # Tomar el valor final en t=1 para cada muestra
            y_final = out[-1]
            return y_final  # forma [batch_size, features]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class StudentODE_new(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone     = NeuralODEModel()
            self.head_clean   = nn.Linear(1, 1)  # Rs
            self.head_distill = nn.Linear(1, 1)  # Rd

        def forward(self, x):
            h  = self.backbone(x)    # [batch,1]
            Rs = self.head_clean(h)   # [batch,1]
            Rd = self.head_distill(h) # [batch,1]
            return Rs, Rd

    def kl_normals_torch(mu0, sigma0, mu1, sigma1):
        """
        KL(N(mu0, sigma0^2) || N(mu1, sigma1^2))
        = log(sigma1/sigma0)
        + [sigma0^2 + (mu0 - mu1)^2] / (2 * sigma1^2)
        - 0.5
        Todas las operaciones son torch ops, para conservar gradientes.
        """
        return (
            torch.log(sigma1 / sigma0)
            + (sigma0**2 + (mu0 - mu1)**2) / (2 * sigma1**2)
            - 0.5
        )

    class New_loss(nn.Module):
        def __init__(self):
            super(New_loss, self).__init__()
        def forward(self, pred ,teacher , target ):

            resta_1 = pred - target
            resta_2 = teacher - target
            # KL
            mu0     = resta_1.mean()
            sigma0  = resta_1.std(unbiased=False)

            mu1     = resta_2.mean()
            sigma1  = resta_2.std(unbiased=False)

            # KL con torch ops
            kl_value = kl_normals_torch(mu0, sigma0, mu1, sigma1)
            return kl_value

    def train_student_ode(
        model, X_train, y_train, r_t_train,
        X_val,   y_val,   r_t_val,
        batch_size=128, epochs=20, lr=1e-3
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        # DataLoaders
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1,1), dtype=torch.float32),
            torch.tensor(r_t_train,    dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterio = New_loss()

        for epoch in range(1, epochs+1):
            model.train()
            tot_loss = 0.0

            for Xb, tb, rtb in train_loader:
                Xb, tb, rtb = Xb.to(device), tb.to(device), rtb.to(device)
                optimizer.zero_grad()

                Rs, Rd = model(Xb)  # Rs: [B,1]


                loss = criterio(Rs, rtb, tb)  # Usando la nueva función de pérdida

                loss.backward()
                optimizer.step()

                tot_loss += loss.item() * Xb.size(0)

            tot_loss /= len(train_ds)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {tot_loss:.6f}")

        return model


    ds = load_data_clean()
    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)
    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    import csv

    # Leer validación
    r_t_val = []
    with open('teacher_results/val_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
            fila_convertida = [float(x) for x in fila]
            r_t_val.append(fila_convertida)

    

    # Leer entrenamiento
    r_t_train = []
    with open('teacher_results/train_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            fila_convertida = [float(x) for x in fila]
            r_t_train.append(fila_convertida)


    student = StudentODE_new()

    student_trained = train_student_ode(
        student,
        X_train, y_train, r_t_train,
        X_test,   y_test,   r_t_val,
        batch_size=128,
        epochs=20,
        lr=0.001
    )

    student.eval()
    Rs_pred, Rd_pred = student(torch.tensor(X_test, dtype=torch.float32).to(device))
    # Rd_pred es tu salida distill, compárala con y_val:
    results =  Rs_pred.cpu().detach().numpy().flatten()

    # metrics

    mae = mean_absolute_error(y_test, results)
    mse = mean_squared_error(y_test, results)
    smape_value = smape(y_test, results)

    print("Metrics for Nueral 3% Model:") 
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)
    torch.save(student.state_dict(), "Models_neural/modelo_3_dest_kl.pth")

    return True


def train_LSTM_KL():
    """Train LSTM 3% distilled using a new Loss function based on KL Divergence"""
    os.makedirs('Models_lstm', exist_ok=True)

    ds = load_data_clean()

    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)
    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)

    # Leer datos teacher
    r_t_val = []
    with open('teacher_results/val_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
            fila_convertida = [float(x) for x in fila]
            r_t_val.append(fila_convertida)

    # Leer entrenamiento
    r_t_train = []
    with open('teacher_results/train_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            fila_convertida = [float(x) for x in fila]
            r_t_train.append(fila_convertida)

    t_train = np.array(y_train).reshape(-1,1)
    t_val   = np.array(y_val).reshape(-1,1)


    y_true_train = np.concatenate([t_train, r_t_train], axis=1)
    y_true_val   = np.concatenate([t_val,   r_t_val],   axis=1)


    @tf.keras.utils.register_keras_serializable(package="CustomModels")
    class Student_KL(Model):
        def __init__(
            self,
            window_size: int,
            n_features: int,
            lstm_units: int,
            c_kl: float = 1.0,      # ahora pondera la KL
            eps_sigma: float = 1e-6, # para evitar nan en sigma
            **kwargs
        ):
            super().__init__(**kwargs)
            self.window_size = window_size
            self.n_features  = n_features
            self.lstm_units  = lstm_units
            self.c_kl        = c_kl
            self.eps_sigma   = eps_sigma  # para evitar nan en sigma
            # métricas
            self.tor_metric  = tf.keras.metrics.Mean(name="tor_loss")
            self.kl_metric   = tf.keras.metrics.Mean(name="kl_loss")

            # capas
            self.lstm = LSTM(lstm_units, input_shape=(window_size, n_features), name="lstm_layer")
            self.dense_clean   = Dense(1, name="clean_output")    # Rs
            self.dense_teacher = Dense(1, name="teacher_output")  # Rd (opcional usarla)

        def call(self, inputs, training=False):
            x = self.lstm(inputs)
            return self.dense_clean(x), self.dense_teacher(x)


        # ---------- KL entre normales (tensors TF) ----------
        def kl_normals_tf(self, mu0, sigma0, mu1, sigma1):
            # sigma0, sigma1 scalars > 0
            sigma0 = tf.maximum(sigma0, self.eps_sigma)
            sigma1 = tf.maximum(sigma1, self.eps_sigma)

            log_term = tf.math.log(sigma1 / sigma0)
            frac = (sigma0 ** 2 + (mu0 - mu1) ** 2) / (2.0 * sigma1 ** 2)
            kl = log_term + frac - 0.5
            return kl

        def compute_tor_loss(self, t, r_t_gt, Rs, epsilon):
            err = tf.abs(t - r_t_gt)
            clean_loss   = tf.square(Rs - t)
            outlier_loss = tf.sqrt(tf.square(Rs - r_t_gt) + 1e-6)
            return tf.where(err < epsilon, clean_loss, outlier_loss)

        # ---------- train_step (usa KL en lugar de L1 para distill) ----------
        def train_step(self, data):
            x, y = data
            t      = y[:, 0:1]   # etiqueta real (N,1)
            r_t_gt = y[:, 1:2]   # pred teacher (N,1)

            # estimar epsilon (del batch)
            with tf.GradientTape() as tape:
                Rs, Rd = self(x, training=True)            # (N,1), (N,1)
                
                # KL between residual distributions
                resid_student = tf.reshape(Rs - t, [-1])   # (N,)
                resid_teacher = tf.reshape(r_t_gt - t, [-1])

                mu0 = tf.reduce_mean(resid_student)
                mu1 = tf.reduce_mean(resid_teacher)
                sigma0 = tf.math.reduce_std(resid_student)   # population std
                sigma1 = tf.math.reduce_std(resid_teacher)

                L_kl = self.kl_normals_tf(mu0, sigma0, mu1, sigma1)
                # opcional: tomar mean si L_kl fuera vector (aquí es escalar)

                #loss = self.c_tor * L_tor + self.c_kl * L_kl
                loss =  self.c_kl * L_kl


            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # actualizar métricas
            self.compiled_metrics.update_state(t, Rs)
            self.kl_metric.update_state(L_kl)

            out = {
                "loss": loss,
                "tor_loss": self.tor_metric.result(),
                "kl_loss": self.kl_metric.result(),
            }
            for m in self.metrics:
                out[m.name] = m.result()
            return out

        def test_step(self, data):
            x, y = data
            t      = y[:, 0:1]
            r_t_gt = y[:, 1:2]
            

            Rs, Rd = self(x, training=False)

            resid_student = tf.reshape(Rs - t, [-1])
            resid_teacher = tf.reshape(r_t_gt - t, [-1])
            mu0 = tf.reduce_mean(resid_student)
            mu1 = tf.reduce_mean(resid_teacher)
            sigma0 = tf.math.reduce_std(resid_student)
            sigma1 = tf.math.reduce_std(resid_teacher)
            L_kl = self.kl_normals_tf(mu0, sigma0, mu1, sigma1)

            loss = self.c_kl * L_kl

            self.compiled_metrics.update_state(t, Rs)
            return {
                "loss": loss,
                "kl_loss": L_kl,
                **{m.name: m.result() for m in self.metrics}
            }

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "window_size": self.window_size,
                "n_features":  self.n_features,
                "lstm_units":  self.lstm_units,
                "c_kl":        self.c_kl,
                "eps_sigma":   self.eps_sigma,
            }

        @classmethod
        def from_config(cls, config):
            return cls(
                window_size=config.pop("window_size"),
                n_features=config.pop("n_features"),
                lstm_units=config.pop("lstm_units"),
                c_kl=config.pop("c_kl", 1.0),
                eps_sigma=config.pop("eps_sigma", 1e-6),
                **config
            )


    student = Student_KL(
        window_size=15,
        n_features=1,
        lstm_units=50
    )

    student.compile(optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])


    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = student.fit(
        X_train, y_true_train,
        validation_data=(X_val, y_true_val),
        epochs=20,
        batch_size=128,
        callbacks=[es]
    )


    Rs_output, Rd_output = student.predict(X_val)
    lstm_destillated = Rs_output.flatten() 

    # Metrics
    print("Metrics for LSTM Distillated Model:")

    mae = mean_absolute_error(y_val, lstm_destillated)
    mse = mean_squared_error(y_val, lstm_destillated)
    smape_val = smape(y_val, lstm_destillated)

    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_val)

    # Save the model
    student.save('Models_lstm/lstm_healthcare_model_3_distillated_kl.keras')
    return True



# New training estimate the variance
def new_train_Neural_KL():
    """Train Neural Network 3% distilled using a new Loss function based on KL Divergence"""

    os.makedirs('Models_neural', exist_ok=True)

    # Neural ODE Model
    class ODEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 50),
                nn.Tanh(),
                nn.Linear(50, 1)
            )

        def forward(self, t, x):
            return self.net(x)

    class NeuralODEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.odefunc = ODEFunc()
        
        def forward(self, x):
            # x: [batch_size, seq_len, features]
            device = x.device
            seq_len = x.shape[1]
            # Crear t en el dispositivo correcto
            t = torch.linspace(0, 1, seq_len, device=device)
            # Estado inicial para cada muestra: último valor de la secuencia
            # Si features=1, x[:, -1, :] es [batch_size, 1]
            y0 = x[:, -1, :]
            # Integrar en batch: devuelve [len(t), batch_size, features]
            out = odeint(self.odefunc, y0, t, method='rk4')
            # Tomar el valor final en t=1 para cada muestra
            y_final = out[-1]
            return y_final  # forma [batch_size, features]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class StudentODE_new(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone     = NeuralODEModel()
            self.head_clean   = nn.Linear(1, 1)  # Rs
            self.head_distill = nn.Linear(1, 1)  # Rd

        def forward(self, x):
            h  = self.backbone(x)    # [batch,1]
            Rs = self.head_clean(h)   # [batch,1]
            Rd = self.head_distill(h) # [batch,1]
            return Rs, Rd

    def kl_normals_torch(mu0, sigma0, mu1, sigma1):
        """
        KL(N(mu0, sigma0^2) || N(mu1, sigma1^2))
        = log(sigma1/sigma0)
        + [sigma0^2 + (mu0 - mu1)^2] / (2 * sigma1^2)
        - 0.5
        Todas las operaciones son torch ops, para conservar gradientes.
        """
        return (
            torch.log(sigma1 / sigma0)
            + (sigma0**2 + (mu0 - mu1)**2) / (2 * sigma1**2)
            - 0.5
        )

    class New_loss(nn.Module):
        def __init__(self):
            super(New_loss, self).__init__()
        def forward(self, pred ,teacher , target ):

            resta_1 = pred - target
            resta_2 = teacher - target
            # KL
            mu0     = resta_1.mean()
            sigma0  = resta_1.std(unbiased=False)

            mu1     = resta_2.mean()
            sigma1  = resta_2.std(unbiased=False)

            # KL con torch ops
            kl_value = kl_normals_torch(mu0, sigma0, mu1, sigma1)
            return kl_value

    def train_student_ode(
        model, X_train, y_train, r_t_train,
        X_val,   y_val,   r_t_val,
        batch_size=128, epochs=20, lr=1e-3
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        # DataLoaders
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1,1), dtype=torch.float32),
            torch.tensor(r_t_train,    dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterio = New_loss()

        for epoch in range(1, epochs+1):
            model.train()
            tot_loss = 0.0

            for Xb, tb, rtb in train_loader:
                Xb, tb, rtb = Xb.to(device), tb.to(device), rtb.to(device)
                optimizer.zero_grad()

                Rs, Rd = model(Xb)  # Rs: [B,1]


                loss = criterio(Rs, rtb, tb)  # Usando la nueva función de pérdida

                loss.backward()
                optimizer.step()

                tot_loss += loss.item() * Xb.size(0)

            tot_loss /= len(train_ds)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {tot_loss:.6f}")

        return model


    ds = load_data_clean()
    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)
    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    import csv

    # Leer validación
    r_t_val = []
    with open('teacher_results/val_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
            fila_convertida = [float(x) for x in fila]
            r_t_val.append(fila_convertida)

    

    # Leer entrenamiento
    r_t_train = []
    with open('teacher_results/train_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            fila_convertida = [float(x) for x in fila]
            r_t_train.append(fila_convertida)


    student = StudentODE_new()

    student_trained = train_student_ode(
        student,
        X_train, y_train, r_t_train,
        X_test,   y_test,   r_t_val,
        batch_size=128,
        epochs=20,
        lr=0.001
    )

    student.eval()
    Rs_pred, Rd_pred = student(torch.tensor(X_test, dtype=torch.float32).to(device))
    # Rd_pred es tu salida distill, compárala con y_val:
    results =  Rs_pred.cpu().detach().numpy().flatten()

    # metrics

    mae = mean_absolute_error(y_test, results)
    mse = mean_squared_error(y_test, results)
    smape_value = smape(y_test, results)

    print("Metrics for Nueral 3% Model:") 
    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_value)
    torch.save(student.state_dict(), "Models_neural/modelo_3_dest_kl.pth")

    return True


def new_train_LSTM_KL():
    """Train LSTM 3% distilled using a new Loss function based on KL Divergence"""
    os.makedirs('Models_lstm', exist_ok=True)

    ds = load_data_clean()

    X, y = create_windows_from_sequences(ds, window_size=15, horizon=1)
    X_3, y_3 = sample_fraction(X, y, 0.03, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_3, y_3, test_size=0.2, shuffle=True, random_state=42)

    # Leer datos teacher
    r_t_val = []
    with open('teacher_results/val_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # Si tus datos eran numéricos, conviértelos, por ejemplo, a float:
            fila_convertida = [float(x) for x in fila]
            r_t_val.append(fila_convertida)

    # Leer entrenamiento
    r_t_train = []
    with open('teacher_results/train_teacher.csv', newline='') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            fila_convertida = [float(x) for x in fila]
            r_t_train.append(fila_convertida)

    t_train = np.array(y_train).reshape(-1,1)
    t_val   = np.array(y_val).reshape(-1,1)


    y_true_train = np.concatenate([t_train, r_t_train], axis=1)
    y_true_val   = np.concatenate([t_val,   r_t_val],   axis=1)


    @tf.keras.utils.register_keras_serializable(package="CustomModels")
    class Student_KL(Model):
        def __init__(
            self,
            window_size: int,
            n_features: int,
            lstm_units: int,
            c_kl: float = 1.0,      # ahora pondera la KL
            eps_sigma: float = 1e-6, # para evitar nan en sigma
            **kwargs
        ):
            super().__init__(**kwargs)
            self.window_size = window_size
            self.n_features  = n_features
            self.lstm_units  = lstm_units
            self.c_kl        = c_kl
            self.eps_sigma   = eps_sigma  # para evitar nan en sigma
            # métricas
            self.tor_metric  = tf.keras.metrics.Mean(name="tor_loss")
            self.kl_metric   = tf.keras.metrics.Mean(name="kl_loss")

            # capas
            self.lstm = LSTM(lstm_units, input_shape=(window_size, n_features), name="lstm_layer")
            self.dense_clean   = Dense(1, name="clean_output")    # Rs
            self.dense_teacher = Dense(1, name="teacher_output")  # Rd (opcional usarla)

        def call(self, inputs, training=False):
            x = self.lstm(inputs)
            return self.dense_clean(x), self.dense_teacher(x)


        # ---------- KL entre normales (tensors TF) ----------
        def kl_normals_tf(self, mu0, sigma0, mu1, sigma1):
            # sigma0, sigma1 scalars > 0
            sigma0 = tf.maximum(sigma0, self.eps_sigma)
            sigma1 = tf.maximum(sigma1, self.eps_sigma)

            log_term = tf.math.log(sigma1 / sigma0)
            frac = (sigma0 ** 2 + (mu0 - mu1) ** 2) / (2.0 * sigma1 ** 2)
            kl = log_term + frac - 0.5
            return kl

        def compute_tor_loss(self, t, r_t_gt, Rs, epsilon):
            err = tf.abs(t - r_t_gt)
            clean_loss   = tf.square(Rs - t)
            outlier_loss = tf.sqrt(tf.square(Rs - r_t_gt) + 1e-6)
            return tf.where(err < epsilon, clean_loss, outlier_loss)

        # ---------- train_step (usa KL en lugar de L1 para distill) ----------
        def train_step(self, data):
            x, y = data
            t      = y[:, 0:1]   # etiqueta real (N,1)
            r_t_gt = y[:, 1:2]   # pred teacher (N,1)

            # estimar epsilon (del batch)
            with tf.GradientTape() as tape:
                Rs, Rd = self(x, training=True)            # (N,1), (N,1)
                
                # KL between residual distributions
                resid_student = tf.reshape(Rs - t, [-1])   # (N,)
                resid_teacher = tf.reshape(r_t_gt - t, [-1])

                mu0 = tf.reduce_mean(resid_student)
                mu1 = tf.reduce_mean(resid_teacher)
                sigma0 = tf.math.reduce_std(resid_student)   # population std
                sigma1 = tf.math.reduce_std(resid_teacher)

                L_kl = self.kl_normals_tf(mu0, sigma0, mu1, sigma1)
                # opcional: tomar mean si L_kl fuera vector (aquí es escalar)

                #loss = self.c_tor * L_tor + self.c_kl * L_kl
                loss =  self.c_kl * L_kl


            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # actualizar métricas
            self.compiled_metrics.update_state(t, Rs)
            self.kl_metric.update_state(L_kl)

            out = {
                "loss": loss,
                "tor_loss": self.tor_metric.result(),
                "kl_loss": self.kl_metric.result(),
            }
            for m in self.metrics:
                out[m.name] = m.result()
            return out

        def test_step(self, data):
            x, y = data
            t      = y[:, 0:1]
            r_t_gt = y[:, 1:2]
            

            Rs, Rd = self(x, training=False)

            resid_student = tf.reshape(Rs - t, [-1])
            resid_teacher = tf.reshape(r_t_gt - t, [-1])
            mu0 = tf.reduce_mean(resid_student)
            mu1 = tf.reduce_mean(resid_teacher)
            sigma0 = tf.math.reduce_std(resid_student)
            
            sigma1 = tf.math.reduce_std(resid_teacher)
            L_kl = self.kl_normals_tf(mu0, sigma0, mu1, sigma1)

            loss = self.c_kl * L_kl

            self.compiled_metrics.update_state(t, Rs)
            return {
                "loss": loss,
                "kl_loss": L_kl,
                **{m.name: m.result() for m in self.metrics}
            }

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "window_size": self.window_size,
                "n_features":  self.n_features,
                "lstm_units":  self.lstm_units,
                "c_kl":        self.c_kl,
                "eps_sigma":   self.eps_sigma,
            }

        @classmethod
        def from_config(cls, config):
            return cls(
                window_size=config.pop("window_size"),
                n_features=config.pop("n_features"),
                lstm_units=config.pop("lstm_units"),
                c_kl=config.pop("c_kl", 1.0),
                eps_sigma=config.pop("eps_sigma", 1e-6),
                **config
            )


    student = Student_KL(
        window_size=15,
        n_features=1,
        lstm_units=50
    )

    # primera estimación del sigma
    xi = t_train - r_t_train
    xi = xi.reshape(-1)  # asegurar shape 1D
    mad = np.median(np.abs(xi - np.median(xi)))
    sigma = 1.4826 * mad 

    #Añadir el calculo del mu para la estimación
    # 

    
    student.compile(optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])


    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = student.fit(
        X_train, y_true_train,
        validation_data=(X_val, y_true_val),
        epochs=20,
        batch_size=128,
        callbacks=[es]
    )


    Rs_output, Rd_output = student.predict(X_val)
    lstm_destillated = Rs_output.flatten() 

    # Metrics
    print("Metrics for LSTM Distillated Model:")

    mae = mean_absolute_error(y_val, lstm_destillated)
    mse = mean_squared_error(y_val, lstm_destillated)
    smape_val = smape(y_val, lstm_destillated)

    print("MAE:", mae)
    print("MSE:", mse)
    print("SMAPE:", smape_val)

    # Save the model
    student.save('Models_lstm/lstm_healthcare_model_3_distillated_kl.keras')
    return True
