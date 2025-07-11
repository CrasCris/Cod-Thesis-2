import pytest
import random
from Time_MoE.time_moe.datasets.time_moe_dataset import TimeMoEDataset

def test_hospital():
    ds = TimeMoEDataset('Time-300B/healthcare/hospital', normalization_method='max')
    print("Numero de secuencias:", len(ds))
    assert len(ds) == 767

def test_covid():
    ds = TimeMoEDataset('Time-300B/healthcare/covid_deaths', normalization_method='max')
    assert len(ds) == 2
    print("Numero de secuencias:", len(ds))

def test_births():
    ds = TimeMoEDataset('Time-300B/healthcare/us_births', normalization_method='max')
    assert len(ds) == 1
    print("Numero de secuencias:", len(ds))

def test_project_tycho():
    ds = TimeMoEDataset('Time-300B/healthcare/project_tycho', normalization_method='max')
    assert len(ds) == 588
    print("Numero de secuencias:", len(ds))

def test_cdc1():
    ds = TimeMoEDataset('Time-300B/healthcare/cdc_fluview_ilinet', normalization_method='max')
    assert len(ds) == 286
    print("Numero de secuencias:", len(ds))

def test_cdc2():
    ds = TimeMoEDataset('Time-300B/healthcare/cdc_fluview_who_nrevss', normalization_method='max')
    assert len(ds) == 108
    print("Numero de secuencias:", len(ds))


