import subprocess
import pandas as pd
from sodapy import Socrata

def clean_data_covid(datos_covid: pd.DataFrame):
    # Estandarizo nombres de columna en minúsculas
    datos_covid = datos_covid.rename(columns=str.lower)
    # Ahora uso 'departamento' y 'fecha_inicio_sintomas'
    
    # 1) Parseo fechas
    datos_covid['fecha_de_inicio'] = pd.to_datetime(
        datos_covid['fecha_inicio_sintomas'],
        errors='coerce'
    )
    datos_covid = datos_covid.dropna(subset=['fecha_de_inicio'])
    
    # 2) Frecuencias totales
    casos_inicio = (
        datos_covid['fecha_de_inicio']
        .value_counts()
        .rename_axis('fecha')
        .reset_index(name='frecuencia')
        .sort_values('fecha')
    )
    
    # 3) Frecuencias solo Santander (sin distinguir mayúsculas)
    mask = datos_covid['departamento_nom'].str.strip().str.upper() == 'SANTANDER'
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
    
    # 5) Merge y rellenar NaN a 0
    df_total = pd.DataFrame({'fecha': fechas_completas})
    df_total = df_total.merge(casos_inicio, on='fecha', how='left')
    df_total['frecuencia'] = df_total['frecuencia'].fillna(0).astype(int)
    df_total.to_csv('serie_completa.csv', index=False)
    
    df_sant = pd.DataFrame({'fecha': fechas_completas})
    df_sant = df_sant.merge(casos_santander, on='fecha', how='left')
    df_sant['frecuencia'] = df_sant['frecuencia'].fillna(0).astype(int)
    df_sant.to_csv('serie.csv', index=False)



def download_time300b():
    """
    Descargar los datos de Time-300B usando la CLI de Hugging Face. 
    """
    cmd = [
        "huggingface-cli", "download",
        "Maple728/Time-300B",
        "--include", "healthcare/*",
        "--repo-type", "dataset",
        "--local-dir", "./Time-300B"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Descarga completada exitosamente vía CLI.")
    else:
        print("Error durante la descarga:")
        print(result.stderr)


if __name__ == "__main__":
    download_time300b()
