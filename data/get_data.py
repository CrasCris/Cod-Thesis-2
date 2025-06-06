import random
from Time_MoE.time_moe.datasets.time_moe_dataset import TimeMoEDataset


ds = TimeMoEDataset('data/Time-300B/healthcare/cdc_fluview_ilinet',
                    normalization_method='max')
# Selecciona una secuencia aleatoria
seq_idx = random.randint(0, len(ds) - 1)
seq = ds[seq_idx]
print("Secuencia seleccionada:", seq)
