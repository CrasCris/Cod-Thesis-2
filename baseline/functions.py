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