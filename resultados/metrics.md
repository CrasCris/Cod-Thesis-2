# Tablas de comparación


## ANTIOQUIA

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 70.223 | 16.392 | 73.684 | 75.614 | 16.708 | 32.606 | 14.736 |
| MSE | 1,594,751 | 112,948 | 1,595,543 | 1,597,376 | 112,951 | 113,034 | 103,301 |
| MAE | 673.349 | 142.055 | 674.501 | 675.500 | 142.063 | 143.414 | 142.182 |


## BOGOTA

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 78.872 | 10.981 | 83.509 | 84.975 | 11.069 | 18.189 | 11.063 |
| MSE | 3,462,378 | 102,694 | 3,463,608 | 3,466,158 | 102,694 | 102,789 | 109,406 |
| MAE | 991.219 | 135.187 | 992.735 | 993.854 | 135.192 | 136.313 | 140.168 |


## BOYACA

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 59.258 | 30.711 | 59.943 | 63.534 | 32.489 | 58.354 | 26.292 |
| MSE | 36,842 | 3,062.270 | 37,044 | 37,321 | 3,062.834 | 3,115.453 | 2,904.978 |
| MAE | 88.400 | 22.816 | 89.356 | 89.969 | 22.844 | 23.916 | 22.940 |

## CALI

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 66.626 | 16.720 | 69.485 | 71.084 | 17.331 | 37.004 | 14.777 |
| MSE | 361,973 | 11,318 | 362,385 | 363,235 | 11,319 | 11,395 | 13,714 |
| MAE | 283.635 | 46.889 | 284.805 | 285.680 | 46.901 | 48.563 | 49.645 |



## CUNDINAMARCA

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 64.232 | 23.875 | 65.469 | 68.682 | 25.057 | 47.317 | 20.458 |
| MSE | 243,006 | 11,525 | 243,346 | 244,028 | 11,526 | 11,590 | 11,756 |
| MAE | 229.842 | 44.518 | 230.818 | 231.662 | 44.540 | 45.722 | 45.829 |



## MEDELLIN

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 67.765 | 18.364 | 70.441 | 72.191 | 18.949 | 37.641 | 15.884 |
| MSE | 534,572 | 35,917 | 535,072 | 536,194 | 35,919 | 35,997 | 35,171 |
| MAE | 386.080 | 80.895 | 387.140 | 388.033 | 80.909 | 82.238 | 82.473 |



## SANTANDER

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 65.316 | 23.757 | 66.796 | 69.754 | 24.871 | 47.188 | 21.134 |
| MSE | 198,829 | 10,837 | 199,150 | 199,779 | 10,838 | 10,903 | 10,502 |
| MAE | 206.701 | 41.757 | 207.729 | 208.573 | 41.773 | 43.016 | 42.120 |



## VALLE

| Métrica | LSTM_kl | NeuralODE_kl | LSTM_distilled | LSTM_normal | NeuralODE_distilled | NeuralODE_normal | Times-MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAPE | 68.612 | 14.932 | 71.736 | 73.070 | 15.400 | 33.120 | 13.449 |
| MSE | 684,666 | 17,975 | 685,196 | 686,344 | 17,976 | 18,056 | 22,428 |
| MAE | 401.568 | 58.646 | 402.759 | 403.646 | 58.656 | 60.211 | 61.242 |

# Tabla acumulada final (KL + BASELINE)

| Group | City | Model | SMAPE | MSE | MAE |
|---|---|---|---:|---:|---:|
| KL | ANTIOQUIA | LSTM_kl | 70.223 | 1,594,751 | 673.349 |
| KL | ANTIOQUIA | NeuralODE_kl | 16.392 | 112,948 | 142.055 |
| BASELINE | ANTIOQUIA | LSTM_distilled | 73.684 | 1,595,543 | 674.501 |
| BASELINE | ANTIOQUIA | LSTM_normal | 75.614 | 1,597,376 | 675.500 |
| BASELINE | ANTIOQUIA | NeuralODE_distilled | 16.708 | 112,951 | 142.063 |
| BASELINE | ANTIOQUIA | NeuralODE_normal | 32.606 | 113,034 | 143.414 |
| BASELINE | ANTIOQUIA | Times-MoE | 14.736 | 103,301 | 142.182 |
| KL | BOGOTA | LSTM_kl | 78.872 | 3,462,378 | 991.219 |
| KL | BOGOTA | NeuralODE_kl | 10.981 | 102,694 | 135.187 |
| BASELINE | BOGOTA | LSTM_distilled | 83.509 | 3,463,608 | 992.735 |
| BASELINE | BOGOTA | LSTM_normal | 84.975 | 3,466,158 | 993.854 |
| BASELINE | BOGOTA | NeuralODE_distilled | 11.069 | 102,694 | 135.192 |
| BASELINE | BOGOTA | NeuralODE_normal | 18.189 | 102,789 | 136.313 |
| BASELINE | BOGOTA | Times-MoE | 11.063 | 109,406 | 140.168 |
| KL | BOYACA | LSTM_kl | 59.258 | 36,842 | 88.400 |
| KL | BOYACA | NeuralODE_kl | 30.711 | 3,062.270 | 22.816 |
| BASELINE | BOYACA | LSTM_distilled | 59.943 | 37,044 | 89.356 |
| BASELINE | BOYACA | LSTM_normal | 63.534 | 37,321 | 89.969 |
| BASELINE | BOYACA | NeuralODE_distilled | 32.489 | 3,062.834 | 22.844 |
| BASELINE | BOYACA | NeuralODE_normal | 58.354 | 3,115.453 | 23.916 |
| BASELINE | BOYACA | Times-MoE | 26.292 | 2,904.978 | 22.940 |
| KL | CALI | LSTM_kl | 66.626 | 361,973 | 283.635 |
| KL | CALI | NeuralODE_kl | 16.720 | 11,318 | 46.889 |
| BASELINE | CALI | LSTM_distilled | 69.485 | 362,385 | 284.805 |
| BASELINE | CALI | LSTM_normal | 71.084 | 363,235 | 285.680 |
| BASELINE | CALI | NeuralODE_distilled | 17.331 | 11,319 | 46.901 |
| BASELINE | CALI | NeuralODE_normal | 37.004 | 11,395 | 48.563 |
| BASELINE | CALI | Times-MoE | 14.777 | 13,714 | 49.645 |
| KL | CUNDINAMARCA | LSTM_kl | 64.232 | 243,006 | 229.842 |
| KL | CUNDINAMARCA | NeuralODE_kl | 23.875 | 11,525 | 44.518 |
| BASELINE | CUNDINAMARCA | LSTM_distilled | 65.469 | 243,346 | 230.818 |
| BASELINE | CUNDINAMARCA | LSTM_normal | 68.682 | 244,028 | 231.662 |
| BASELINE | CUNDINAMARCA | NeuralODE_distilled | 25.057 | 11,526 | 44.540 |
| BASELINE | CUNDINAMARCA | NeuralODE_normal | 47.317 | 11,590 | 45.722 |
| BASELINE | CUNDINAMARCA | Times-MoE | 20.458 | 11,756 | 45.829 |
| KL | MEDELLIN | LSTM_kl | 67.765 | 534,572 | 386.080 |
| KL | MEDELLIN | NeuralODE_kl | 18.364 | 35,917 | 80.895 |
| BASELINE | MEDELLIN | LSTM_distilled | 70.441 | 535,072 | 387.140 |
| BASELINE | MEDELLIN | LSTM_normal | 72.191 | 536,194 | 388.033 |
| BASELINE | MEDELLIN | NeuralODE_distilled | 18.949 | 35,919 | 80.909 |
| BASELINE | MEDELLIN | NeuralODE_normal | 37.641 | 35,997 | 82.238 |
| BASELINE | MEDELLIN | Times-MoE | 15.884 | 35,171 | 82.473 |
| KL | SANTANDER | LSTM_kl | 65.316 | 198,829 | 206.701 |
| KL | SANTANDER | NeuralODE_kl | 23.757 | 10,837 | 41.757 |
| BASELINE | SANTANDER | LSTM_distilled | 66.796 | 199,150 | 207.729 |
| BASELINE | SANTANDER | LSTM_normal | 69.754 | 199,779 | 208.573 |
| BASELINE | SANTANDER | NeuralODE_distilled | 24.871 | 10,838 | 41.773 |
| BASELINE | SANTANDER | NeuralODE_normal | 47.188 | 10,903 | 43.016 |
| BASELINE | SANTANDER | Times-MoE | 21.134 | 10,502 | 42.120 |
| KL | VALLE | LSTM_kl | 68.612 | 684,666 | 401.568 |
| KL | VALLE | NeuralODE_kl | 14.932 | 17,975 | 58.646 |
| BASELINE | VALLE | LSTM_distilled | 71.736 | 685,196 | 402.759 |
| BASELINE | VALLE | LSTM_normal | 73.070 | 686,344 | 403.646 |
| BASELINE | VALLE | NeuralODE_distilled | 15.400 | 17,976 | 58.656 |
| BASELINE | VALLE | NeuralODE_normal | 33.120 | 18,056 | 60.211 |
| BASELINE | VALLE | Times-MoE | 13.449 | 22,428 | 61.242 |
