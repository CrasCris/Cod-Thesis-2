import pandas as pd
import numpy as np
# Importing custom functions

import sys
import os
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)

from data.Time_MoE.time_moe.datasets.time_moe_dataset import TimeMoEDataset
import torch
from torch import nn
from torchdiffeq import odeint

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def create_intervals(df:pd.DataFrame) -> list:
    """
    Divide un DataFrame en 8 intervalos regulares basados en la columna 'Fecha',
    grafica los datos y devuelve los intervalos.
    """
    fecha_min = df['Fecha'].min()
    fecha_max = df['Fecha'].max()
    bins = pd.date_range(start=fecha_min, end=fecha_max, periods=9)  # 9 puntos para generar 8 intervalos
    df['Intervalo'] = pd.cut(df['Fecha'], bins=bins, include_lowest=True, right=False)
    intervalos_ordenados = sorted(df['Intervalo'].dropna().unique(), key=lambda x: x.left)
    return intervalos_ordenados

def create_windows(data, window_size):
    """
    Genera secuencias (ventanas) a partir de la serie de tiempo.
    
    Args:
        data (np.array): Valores escalados de la serie.
        window_size (int): Longitud de la ventana.
        
    Returns:
        tuple: Arrays (x, y) donde x son las secuencias y y el valor a predecir.
    """
    x = []
    y = []
    for i in range(len(data) - window_size):
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(x), np.array(y)

def create_windows_2(data, window_size):
    """
    Genera secuencias (ventanas) a partir de la serie de tiempo.
    
    Args:
        data (np.array): Valores escalados de la serie.
        window_size (int): Longitud de la ventana.
        
    Returns:
        tuple: Arrays (x, y) donde x son las secuencias y y el valor a predecir.
    """
    x = []
    y = []
    ds_np = data.to_numpy()
    for i in range(len(ds_np) - window_size):
        row = [[a] for a in ds_np[i:i+window_size]]
        x.append(row)
        y_val = ds_np[i + window_size]
        y.append([y_val])
    return np.array(x), np.array(y)

def smape(y_true, y_pred, eps=1e-8):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(numerator / (denominator + eps))

def create_windows_multi(data, window_size, horizonte):
    """
    data: array 1D numpy de longitud L
    window_size: tamaño de la ventana de entrada
    horizonte: número de pasos a predecir
    Retorna:
      Xs: np.array de forma (num_samples, window_size)
      ys: np.array de forma (num_samples, horizonte)
    """
    Xs = []
    ys = []
    total = len(data)
    # Para que exista un bloque de window_size + horizonte
    for i in range(total - window_size - horizonte + 1):
        Xs.append(data[i : i + window_size].flatten())
        ys.append(data[i + window_size : i + window_size + horizonte].flatten())
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys

def create_intervals_moe(df):
    fecha_min = df['Fecha'].min()
    fecha_max = df['Fecha'].max()
    bins = pd.date_range(start=fecha_min, end=fecha_max, periods=9)
    df['Intervalo'] = pd.cut(df['Fecha'], bins=bins, include_lowest=True, right=False)
    return sorted(df['Intervalo'].dropna().unique(), key=lambda x: x.left)


def smape_chunked(y_true, y_pred, eps=1e-8, chunk_size=1_000_000):
    """
    Calcula SMAPE de manera eficiente por chunks para evitar usar demasiada memoria de una sola vez.
    Args:
        y_true (np.ndarray): Array de valores reales.
        y_pred (np.ndarray): Array de valores predichos.
        eps (float): Valor pequeño para evitar división por cero.
        chunk_size (int): Número de elementos por chunk para procesar en cada iteración.
    Retorna:
        float: SMAPE en porcentaje.
    """
    # Asegurarnos de que son numpy arrays 1D y de tipo float
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    n = y_true.shape[0]
    assert y_pred.shape[0] == n, f"y_true y y_pred deben tener misma longitud: {n} vs {y_pred.shape[0]}"
    
    total_smape = 0.0
    count = 0
    
    # Procesar en chunks
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yt = y_true[start:end]
        yp = y_pred[start:end]
        
        # Cálculo chunk
        abs_diff = np.abs(yt - yp)
        denom = np.abs(yt) + np.abs(yp)
        
        # Evitar división por cero: considerar solo denom > eps
        mask = denom > eps
        if np.any(mask):
            smape_vals = abs_diff[mask] / (denom[mask] / 2.0)
            total_smape += np.sum(smape_vals)
            count += mask.sum()
        # Para denom <= eps, se omiten (no contribuyen al sumatorio ni al count)
    
    if count == 0:
        return np.nan  # No hay elementos válidos para calcular
    return 100 * (total_smape / count)

def sample_fraction(X, y, fraction, random_state=None, min_samples=1):
    """
    Muestra aleatoriamente una fracción de los datos (X, y).
    
    Args:
        X (np.ndarray): Array de entrada de forma (N, ..., ...), donde N es el número de muestras.
        y (np.ndarray): Array de etiquetas de forma (N, ...) correspondiente a X.
        fraction (float): Fracción de datos a tomar. Debe estar en (0, 1]. Por ejemplo, 0.10 para 10% o 0.03 para 3%.
        random_state (int o None): Semilla para reproducibilidad. Si None, aleatorio.
        min_samples (int): Número mínimo de muestras a retornar si fraction*N < min_samples. Por defecto 1.
    
    Retorna:
        X_sample (np.ndarray): Subconjunto muestreado de X de tamaño aproximadamente floor(N * fraction) o al menos min_samples.
        y_sample (np.ndarray): Subconjunto muestreado de y correspondiente.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.shape[0] == y.shape[0], f"X e y deben tener el mismo número de muestras en la dimensión 0: {X.shape[0]} vs {y.shape[0]}"
    assert 0 < fraction <= 1, f"fraction debe estar en (0, 1], se recibió: {fraction}"
    
    N = X.shape[0]
    # Calcular tamaño de la muestra
    sample_size = int(np.floor(N * fraction))
    # Asegurar al menos min_samples si sample_size es menor
    sample_size = max(sample_size, min_samples) if N > 0 else 0
    sample_size = min(sample_size, N)  # no exceder N
    
    # Generar índices aleatorios sin reemplazo
    rng = np.random.default_rng(random_state)
    indices = rng.choice(N, size=sample_size, replace=False)
    
    return X[indices], y[indices]

def create_windows_from_sequences(sequences, window_size=15, horizon=1):
    """
    Dada una lista de secuencias (numpy arrays 1D), crea ventanas deslizantes:
    - X: array de shape (num_samples, window_size, 1)
    - y: array de shape (num_samples,)
    Cada muestra usa window_size pasos para predecir el siguiente valor (horizon=1).
    """
    X_list = []
    y_list = []
    for seq in sequences:
        # Asegurar numpy array
        arr = np.array(seq).astype(float)
        T = arr.shape[0]
        # Solo si la longitud es mayor que window_size + horizon - 1
        if T >= window_size + horizon:
            for start in range(0, T - window_size - horizon + 1):
                window = arr[start:start+window_size]
                target = arr[start+window_size:start+window_size+horizon]
                # Para horizon=1, target es un array de longitud 1; tomamos el escalar
                X_list.append(window.reshape(window_size, 1))
                y_list.append(target[0] if horizon == 1 else target)
    if len(X_list) == 0:
        return np.empty((0, window_size, 1)), np.empty((0,))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    # Supongamos X tiene forma (N, window_size, 1), y y forma (N,)
    mask_valid = ~np.isnan(X).any(axis=(1,2)) & ~np.isnan(y)
    # Mantener solo muestras sin NaN:
    X_clean = X[mask_valid]
    y_clean = y[mask_valid]
    print("De", X.shape[0], "muestras, quedan", X_clean.shape[0], "sin NaN")

    return X_clean, y_clean

def load_data_clean():
    ds = TimeMoEDataset(data_folder='Time-300B\healthcare',normalization_method='zero')

    verbose = True
    total = len(ds)
    valid_indices = []
    # Iterar y filtrar
    for i in range(total):
        try:
            seq = ds[i]  # seq es numpy.ndarray según comprobaste
        except Exception as e:
            # Si hay error al obtener la secuencia, lo avisamos y saltamos
            if verbose:
                print(f"Advertencia: no se pudo obtener ds[{i}]: {e}")
            continue
        
        # Comprobación: si todos los valores son NaN, lo descartamos
        # seq es numpy.ndarray; cuidado si dims especiales, pero np.isnan funcionará elementwise.
        try:
            if not np.all(np.isnan(seq)):
                valid_indices.append(i)
        except Exception as e:
            # En caso de que seq no sea array puro, convertir primero:
            try:
                arr = np.array(seq)
                if not np.all(np.isnan(arr)):
                    valid_indices.append(i)
            except Exception as e2:
                if verbose:
                    print(f"Error al verificar NaN en secuencia índice {i}: {e2}")
                # Decidir si incluirla o no. Aquí optamos por descartarla:
                continue
    
    valid_count = len(valid_indices)
    if verbose:
        print(f"Secuencias totales en ds: {total}")
        print(f"Secuencias válidas (no todo NaN): {valid_count}")
        print(f"Secuencias descartadas: {total - valid_count}")
        sequences_validas = []

    for idx in valid_indices:
        try:
            sequences_validas.append(ds[idx])
        except Exception as e:
            if verbose:
                print(f"Error al extraer ds[{idx}] después de filtrar: {e}")
            # Podrías decidir saltar o detener. Aquí solo saltamos.
    return sequences_validas


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
    
    

def create_windows_inference(data, window_size=15, horizon=1):
    """
    Crea ventanas deslizantes (X, y) a partir de:
      - data: puede ser
          * un numpy array 1D (una sola serie temporal), o
          * una lista/iterable de numpy arrays 1D (varias series).
    Parámetros:
      - window_size: número de pasos de entrada por ventana X
      - horizon: pasos a predecir (por defecto 1)
    Devuelve:
      - X: array de shape (num_samples, window_size, 1)
      - y: array de shape (num_samples,) si horizon=1,
             o (num_samples, horizon) en otro caso.
    """
    # Si data es un array 1D, lo convertimos en lista de una sola secuencia
    if isinstance(data, np.ndarray) and data.ndim == 1:
        sequences = [data]
    else:
        # Asumimos que data es iterable de arrays 1D
        sequences = [np.asarray(seq).astype(float) for seq in data]

    X_list, y_list = [], []

    for arr in sequences:
        T = arr.shape[0]
        # Solo procesar si hay suficiente longitud
        if T >= window_size + horizon:
            for start in range(T - window_size - horizon + 1):
                window = arr[start : start + window_size]
                target = arr[start + window_size : start + window_size + horizon]
                X_list.append(window.reshape(window_size, 1))
                # Si horizon=1 devolvemos escalar, si no, el vector entero
                y_list.append(target[0] if horizon == 1 else target)

    # Si no se creó ninguna muestra, devolvemos arrays vacíos con las formas correctas
    if not X_list:
        X = np.empty((0, window_size, 1))
        y = np.empty((0,)) if horizon == 1 else np.empty((0, horizon))
        return X, y

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    # Filtrar muestras que contengan NaN en X o en y
    mask_valid = ~np.isnan(X).any(axis=(1,2))
    if horizon == 1:
        mask_valid &= ~np.isnan(y)
    else:
        mask_valid &= ~np.isnan(y).any(axis=1)

    X_clean = X[mask_valid]
    y_clean = y[mask_valid]

    print(f"De {X.shape[0]} muestras, quedan {X_clean.shape[0]} sin NaN")
    return X_clean, y_clean