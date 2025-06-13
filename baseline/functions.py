import pandas as pd
import numpy as np

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
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
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

