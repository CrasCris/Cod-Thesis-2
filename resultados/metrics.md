# Tablas de comparación — métricas por ciudad KL

Formato: cada tabla muestra las métricas (filas) y los modelos (columnas) para esa ciudad.

## ANTIOQUIA

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 70.223 | 16.392 |
| MSE | 1,594,751 | 112,948 |
| MAE | 673.349 | 142.055 |


## BOGOTA

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 78.872 | 10.981 |
| MSE | 3,462,378 | 102,694 |
| MAE | 991.219 | 135.187 |


## BOYACA

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 59.258 | 30.711 |
| MSE | 36,842 | 3062.270 |
| MAE | 88.400 | 22.816 |


## CALI

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 66.626 | 16.720 |
| MSE | 361,973 | 11,318 |
| MAE | 283.635 | 46.889 |


## CUNDINAMARCA

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 64.232 | 23.875 |
| MSE | 243,006 | 11,525 |
| MAE | 229.842 | 44.518 |


## MEDELLIN

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 67.765 | 18.364 |
| MSE | 534,572 | 35,917 |
| MAE | 386.080 | 80.895 |


## SANTANDER

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 65.316 | 23.757 |
| MSE | 198,829 | 10,837 |
| MAE | 206.701 | 41.757 |


## VALLE

| Métrica | LSTM_kl | NeuralODE_kl |
|---|---|---|
| SMAPE | 68.612 | 14.932 |
| MSE | 684,666 | 17,975 |
| MAE | 401.568 | 58.646 |


## Tabla larga (por si prefieres una sola tabla)

| City | Model | SMAPE | MSE | MAE |
|---|---:|---:|---:|---:|
| CALI | NeuralODE_kl | 16.720 | 11317.729 | 46.889 |
| CALI | LSTM_kl | 66.626 | 361972.906 | 283.635 |
| BOGOTA | NeuralODE_kl | 10.981 | 102693.609 | 135.187 |
| BOGOTA | LSTM_kl | 78.872 | 3462378.000 | 991.219 |
| MEDELLIN | NeuralODE_kl | 18.364 | 35917.023 | 80.895 |
| MEDELLIN | LSTM_kl | 67.765 | 534572.312 | 386.080 |
| SANTANDER | NeuralODE_kl | 23.757 | 10837.127 | 41.757 |
| SANTANDER | LSTM_kl | 65.316 | 198828.828 | 206.701 |
| ANTIOQUIA | NeuralODE_kl | 16.392 | 112948.328 | 142.055 |
| ANTIOQUIA | LSTM_kl | 70.223 | 1594750.750 | 673.349 |
| VALLE | NeuralODE_kl | 14.932 | 17975.104 | 58.646 |
| VALLE | LSTM_kl | 68.612 | 684666.500 | 401.568 |
| BOYACA | NeuralODE_kl | 30.711 | 3062.270 | 22.816 |
| BOYACA | LSTM_kl | 59.258 | 36841.602 | 88.400 |
| CUNDINAMARCA | NeuralODE_kl | 23.875 | 11524.630 | 44.518 |
| CUNDINAMARCA | LSTM_kl | 64.232 | 243005.797 | 229.842 |






# Tablas de comparación — métricas por ciudad BASELINE

Formato: cada tabla muestra las métricas (filas) y los modelos (columnas) para esa ciudad.

## ANTIOQUIA

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 73.684 | 75.614 | 16.708 | 32.606 | 14.736 |
| MSE | 1,595,543 | 1,597,376 | 112,951 | 113,034 | 103,301 |
| MAE | 674.501 | 675.500 | 142.063 | 143.414 | 142.182 |


## BOGOTA

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 83.509 | 84.975 | 11.069 | 18.189 | 11.063 |
| MSE | 3,463,608 | 3,466,158 | 102,694 | 102,789 | 109,406 |
| MAE | 992.735 | 993.854 | 135.192 | 136.313 | 140.168 |


## BOYACA

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 59.943 | 63.534 | 32.489 | 58.354 | 26.292 |
| MSE | 37,044 | 37,321 | 3062.834 | 3115.453 | 2904.978 |
| MAE | 89.356 | 89.969 | 22.844 | 23.916 | 22.940 |


## CALI

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 69.485 | 71.084 | 17.331 | 37.004 | 14.777 |
| MSE | 362,385 | 363,235 | 11,319 | 11,395 | 13,714 |
| MAE | 284.805 | 285.680 | 46.901 | 48.563 | 49.645 |


## CUNDINAMARCA

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 65.469 | 68.682 | 25.057 | 47.317 | 20.458 |
| MSE | 243,346 | 244,028 | 11,526 | 11,590 | 11,756 |
| MAE | 230.818 | 231.662 | 44.540 | 45.722 | 45.829 |


## MEDELLIN

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 70.441 | 72.191 | 18.949 | 37.641 | 15.884 |
| MSE | 535,072 | 536,194 | 35,919 | 35,997 | 35,171 |
| MAE | 387.140 | 388.033 | 80.909 | 82.238 | 82.473 |


## SANTANDER

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 66.796 | 69.754 | 24.871 | 47.188 | 21.134 |
| MSE | 199,150 | 199,779 | 10,838 | 10,903 | 10,502 |
| MAE | 207.729 | 208.573 | 41.773 | 43.016 | 42.120 |


## VALLE

| Métrica | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---|---|---|---|---|
| SMAPE | 71.736 | 73.070 | 15.400 | 33.120 | 13.449 |
| MSE | 685,196 | 686,344 | 17,976 | 18,056 | 22,428 |
| MAE | 402.759 | 403.646 | 58.656 | 60.211 | 61.242 |


## Tabla larga (por si prefieres una sola tabla)

| City | Model | SMAPE | MSE | MAE |
|---|---|---:|---:|---:|
| CALI | NeuralODE_distilled | 17.331 | 11,319 | 46.901 |
| CALI | NeuralODE_normal | 37.004 | 11,395 | 48.563 |
| CALI | LSTM_distilled | 69.485 | 362,385 | 284.805 |
| CALI | LSTM_normal | 71.084 | 363,235 | 285.680 |
| CALI | Times-MoE | 14.777 | 13,714 | 49.645 |
| BOGOTA | NeuralODE_distilled | 11.069 | 102,694 | 135.192 |
| BOGOTA | NeuralODE_normal | 18.189 | 102,789 | 136.313 |
| BOGOTA | LSTM_distilled | 83.509 | 3,463,608 | 992.735 |
| BOGOTA | LSTM_normal | 84.975 | 3,466,158 | 993.854 |
| BOGOTA | Times-MoE | 11.063 | 109,406 | 140.168 |
| MEDELLIN | NeuralODE_distilled | 18.949 | 35,919 | 80.909 |
| MEDELLIN | NeuralODE_normal | 37.641 | 35,997 | 82.238 |
| MEDELLIN | LSTM_distilled | 70.441 | 535,072 | 387.140 |
| MEDELLIN | LSTM_normal | 72.191 | 536,194 | 388.033 |
| MEDELLIN | Times-MoE | 15.884 | 35,171 | 82.473 |
| SANTANDER | NeuralODE_distilled | 24.871 | 10,838 | 41.773 |
| SANTANDER | NeuralODE_normal | 47.188 | 10,903 | 43.016 |
| SANTANDER | LSTM_distilled | 66.796 | 199,150 | 207.729 |
| SANTANDER | LSTM_normal | 69.754 | 199,779 | 208.573 |
| SANTANDER | Times-MoE | 21.134 | 10,502 | 42.120 |
| ANTIOQUIA | NeuralODE_distilled | 16.708 | 112,951 | 142.063 |
| ANTIOQUIA | NeuralODE_normal | 32.606 | 113,034 | 143.414 |
| ANTIOQUIA | LSTM_distilled | 73.684 | 1,595,543 | 674.501 |
| ANTIOQUIA | LSTM_normal | 75.614 | 1,597,376 | 675.500 |
| ANTIOQUIA | Times-MoE | 14.736 | 103,301 | 142.182 |
| VALLE | NeuralODE_distilled | 15.400 | 17,976 | 58.656 |
| VALLE | NeuralODE_normal | 33.120 | 18,056 | 60.211 |
| VALLE | LSTM_distilled | 71.736 | 685,196 | 402.759 |
| VALLE | LSTM_normal | 73.070 | 686,344 | 403.646 |
| VALLE | Times-MoE | 13.449 | 22,428 | 61.242 |
| BOYACA | NeuralODE_distilled | 32.489 | 3062.834 | 22.844 |
| BOYACA | NeuralODE_normal | 58.354 | 3115.453 | 23.916 |
| BOYACA | LSTM_distilled | 59.943 | 37,044 | 89.356 |
| BOYACA | LSTM_normal | 63.534 | 37,321 | 89.969 |
| BOYACA | Times-MoE | 26.292 | 2904.978 | 22.940 |
| CUNDINAMARCA | NeuralODE_distilled | 25.057 | 11,526 | 44.540 |
| CUNDINAMARCA | NeuralODE_normal | 47.317 | 11,590 | 45.722 |
| CUNDINAMARCA | LSTM_distilled | 65.469 | 243,346 | 230.818 |
| CUNDINAMARCA | LSTM_normal | 68.682 | 244,028 | 231.662 |
| CUNDINAMARCA | Times-MoE | 20.458 | 11,756 | 45.829 |