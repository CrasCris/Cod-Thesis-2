import random
from Time_MoE.time_moe.datasets.time_moe_dataset import TimeMoEDataset

def load_data():
    ds = TimeMoEDataset('Time-300B\healthcare\hospital',
                    normalization_method='max')
    # Selecciona una secuencia aleatoria
    seq_idx = random.randint(0, len(ds) - 1)
    seq = ds[seq_idx]
    return ds

data = load_data()
print(len(data))