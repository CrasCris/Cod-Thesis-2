#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Ruta a tu CSV de métricas
    csv_path = Path('data/metrics.csv')
    df = pd.read_csv(str(csv_path))

    # Pivot so that 'type' are the x-axis, cities are columns, metrics are top-level
    pivot_df = df.pivot(index='type', columns='city', values=['SMAPE','MSE','MAE'])

    # Plot each metric as its own bar chart
    for metric in ['SMAPE', 'MSE', 'MAE']:
        plt.figure()
        pivot_df[metric].plot(kind='bar')
        plt.title(f'{metric} por Modelo y Ciudad')
        plt.xlabel('Modelo')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    df = pd.read_csv('data/metrics.csv')
    metrics = ['SMAPE', 'MSE', 'MAE']
    for metric in metrics:
        pivot = df.pivot(index='city', columns='type', values=metric)
        pivot = pivot[['NeuralODE_distilled','NeuralODE_normal','LSTM_distilled','LSTM_normal','Times-MoE']]
        pivot.columns = ['Neural-d','Neural','LSTM-d','LSTM', 'TimesMoE']
        pivot = pivot.round(3)
        pivot.index.name = 'Ciudad'
        pivot = pivot.reset_index()

        print(f'Summery table for {metric}:')
        print(pivot)



if __name__ == '__main__':
    main()

